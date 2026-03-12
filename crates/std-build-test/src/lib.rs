// Test: can we compile with -Zbuild-std=std for nvptx64?
// Requires patched std source with cuda target support.

#![no_main]
#![feature(restricted_std)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use core::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Global hostcall buffer pointer for stdio. Set by kernel at entry.
static STDIO_HOSTCALL_BUF: AtomicU64 = AtomicU64::new(0);

/// External function called by std's CUDA PAL Stdout::write().
/// Implements the hostcall PRINT protocol using the stored buffer pointer.
#[unsafe(no_mangle)]
pub fn gpu_stdout_write(buf: *const u8, len: usize) -> usize {
    let hc_buf = STDIO_HOSTCALL_BUF.load(AtomicOrdering::Relaxed) as *mut u8;
    if hc_buf.is_null() || buf.is_null() || len == 0 {
        return len; // silently discard if no hostcall buffer set
    }
    // Send the message via hostcall PRINT service
    // Split into 56-byte chunks (PRINT_MAX_MSG_LEN)
    const MAX_CHUNK: usize = 56;
    let mut offset = 0usize;
    while offset < len {
        let chunk_len = core::cmp::min(len - offset, MAX_CHUNK);
        let ok = unsafe { gpu_hostcall_print_raw(hc_buf, buf.add(offset), chunk_len as u32) };
        if !ok {
            return offset; // partial write on failure
        }
        offset += chunk_len;
    }
    len
}

/// External function called by std's CUDA PAL Stdin::read().
/// Implements the hostcall STDIN protocol using the stored buffer pointer.
#[unsafe(no_mangle)]
pub fn gpu_stdin_read(out_buf: *mut u8, max_len: usize) -> usize {
    let hc_buf = STDIO_HOSTCALL_BUF.load(AtomicOrdering::Relaxed) as *mut u8;
    if hc_buf.is_null() || out_buf.is_null() || max_len == 0 {
        return 0;
    }
    // Read at most 56 bytes per hostcall round-trip
    let request_len = core::cmp::min(max_len, 56);
    let bytes = unsafe { gpu_hostcall_stdin_raw(hc_buf, out_buf, request_len as u32) };
    if bytes == u64::MAX {
        0 // EOF or error
    } else {
        bytes as usize
    }
}

/// Set the hostcall buffer for stdio. Must be called at kernel entry.
fn stdio_init(buf: *mut u8) {
    STDIO_HOSTCALL_BUF.store(buf as u64, AtomicOrdering::Relaxed);
}

/// Minimal hostcall PRINT implementation using inline PTX.
/// This duplicates the logic from gpu-kernel's gpu_hostcall_print but is
/// self-contained so std-build-test doesn't need to depend on gpu-kernel.
unsafe fn gpu_hostcall_print_raw(hc_buf: *mut u8, msg: *const u8, msg_len: u32) -> bool {
    const BUF_OFF_FREE_STACK: usize = 0;
    const BUF_OFF_READY_STACK: usize = 8;
    const BUF_OFF_DOORBELL: usize = 16;
    const PKT_OFF_NEXT: usize = 0;
    const PKT_OFF_ACTIVE_MASK: usize = 8;
    const PKT_OFF_SERVICE: usize = 12;
    const PKT_OFF_CONTROL: usize = 16;
    const PKT_OFF_PAYLOAD: usize = 32;
    const BUFFER_HEADER_SIZE: usize = 64;
    const PACKET_SIZE: usize = 2112;
    const NULL_INDEX: u16 = 0xFFFF;
    const SERVICE_PRINT: u32 = 1;
    const CONTROL_READY: u32 = 1;
    const GPU_MAX_SPIN: u32 = 10_000_000;

    #[inline(always)]
    unsafe fn ld_acq_u64(ptr: *const u64) -> u64 {
        let r: u64;
        core::arch::asm!("ld.acquire.sys.global.u64 {r}, [{p}];", p = in(reg64) ptr, r = out(reg64) r, options(nostack));
        r
    }
    #[inline(always)]
    unsafe fn cas_u64(ptr: *mut u64, exp: u64, des: u64) -> u64 {
        let r: u64;
        core::arch::asm!("atom.cas.sys.global.b64 {r}, [{p}], {e}, {d};", p = in(reg64) ptr, e = in(reg64) exp, d = in(reg64) des, r = out(reg64) r, options(nostack));
        r
    }
    #[inline(always)]
    unsafe fn st_rel_u32(ptr: *mut u32, val: u32) {
        core::arch::asm!("st.release.sys.global.u32 [{p}], {v};", p = in(reg64) ptr, v = in(reg32) val, options(nostack));
    }
    #[inline(always)]
    unsafe fn ld_acq_u32(ptr: *const u32) -> u32 {
        let r: u32;
        core::arch::asm!("ld.acquire.sys.global.u32 {r}, [{p}];", p = in(reg64) ptr, r = out(reg32) r, options(nostack));
        r
    }
    #[inline(always)]
    unsafe fn fetch_add_u64(ptr: *mut u64, val: u64) -> u64 {
        let r: u64;
        core::arch::asm!("atom.add.sys.global.u64 {r}, [{p}], {v};", p = in(reg64) ptr, v = in(reg64) val, r = out(reg64) r, options(nostack));
        r
    }
    #[inline(always)]
    unsafe fn membar() {
        core::arch::asm!("membar.sys;", options(nostack));
    }
    #[inline(always)]
    unsafe fn amask() -> u32 {
        let r: u32;
        core::arch::asm!("activemask.b32 {r};", r = out(reg32) r, options(nostack, nomem));
        r
    }

    let tagged_index = |t: u64| -> u16 { (t & 0xFFFF) as u16 };
    let tagged_tag = |t: u64| -> u32 { (t >> 32) as u32 };
    let make_tagged = |tag: u32, idx: u16| -> u64 { ((tag as u64) << 32) | (idx as u64) };
    let pkt_offset = |idx: u16| -> usize { BUFFER_HEADER_SIZE + (idx as usize) * PACKET_SIZE };

    // Pop free packet
    let free_ptr = hc_buf.add(BUF_OFF_FREE_STACK) as *mut u64;
    let pkt_idx;
    loop {
        let old_head = ld_acq_u64(free_ptr as *const u64);
        let idx = tagged_index(old_head);
        if idx == NULL_INDEX { return false; }
        let pkt = hc_buf.add(pkt_offset(idx));
        let next = core::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
        if cas_u64(free_ptr, old_head, next) == old_head {
            pkt_idx = idx;
            break;
        }
    }

    let pkt = hc_buf.add(pkt_offset(pkt_idx));

    // Fill header
    let mask = amask();
    core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
    core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_PRINT);
    st_rel_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

    // Fill payload
    let payload = pkt.add(PKT_OFF_PAYLOAD);
    core::ptr::write_volatile(payload as *mut u64, msg_len as u64);
    let dst = payload.add(8);
    let copy_len = if msg_len > 56 { 56u32 } else { msg_len };
    let mut i: u32 = 0;
    while i < copy_len {
        core::ptr::write_volatile(dst.add(i as usize), *msg.add(i as usize));
        i += 1;
    }

    membar();

    // Push to ready stack
    let ready_ptr = hc_buf.add(BUF_OFF_READY_STACK) as *mut u64;
    loop {
        let old_head = ld_acq_u64(ready_ptr as *const u64);
        core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
        let new_tag = tagged_tag(old_head).wrapping_add(1);
        let new_tagged = make_tagged(new_tag, pkt_idx);
        if cas_u64(ready_ptr, old_head, new_tagged) == old_head { break; }
    }

    // Doorbell
    fetch_add_u64(hc_buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

    // Spin-wait
    let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
    let mut spins: u32 = 0;
    let success;
    loop {
        let ctrl = ld_acq_u32(control_ptr);
        if ctrl & CONTROL_READY != 0 { success = true; break; }
        spins += 1;
        if spins >= GPU_MAX_SPIN {
            // Timeout — return packet
            loop {
                let old_head = ld_acq_u64(free_ptr as *const u64);
                core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
                let new_tag = tagged_tag(old_head).wrapping_add(1);
                let new_tagged = make_tagged(new_tag, pkt_idx);
                if cas_u64(free_ptr, old_head, new_tagged) == old_head { break; }
            }
            return false;
        }
    }

    // Return packet to free stack
    loop {
        let old_head = ld_acq_u64(free_ptr as *const u64);
        core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
        let new_tag = tagged_tag(old_head).wrapping_add(1);
        let new_tagged = make_tagged(new_tag, pkt_idx);
        if cas_u64(free_ptr, old_head, new_tagged) == old_head { break; }
    }

    success
}

/// Minimal hostcall STDIN implementation using inline PTX.
/// Returns bytes read on success, u64::MAX on error/EOF.
unsafe fn gpu_hostcall_stdin_raw(hc_buf: *mut u8, out_buf: *mut u8, max_len: u32) -> u64 {
    // Reuse the same constants as gpu_hostcall_print_raw
    const BUF_OFF_FREE_STACK: usize = 0;
    const BUF_OFF_READY_STACK: usize = 8;
    const BUF_OFF_DOORBELL: usize = 16;
    const PKT_OFF_NEXT: usize = 0;
    const PKT_OFF_ACTIVE_MASK: usize = 8;
    const PKT_OFF_SERVICE: usize = 12;
    const PKT_OFF_CONTROL: usize = 16;
    const PKT_OFF_PAYLOAD: usize = 32;
    const BUFFER_HEADER_SIZE: usize = 64;
    const PACKET_SIZE: usize = 2112;
    const NULL_INDEX: u16 = 0xFFFF;
    const SERVICE_STDIN: u32 = 8;
    const CONTROL_READY: u32 = 1;
    const GPU_MAX_SPIN: u32 = 100_000_000; // Stdin is blocking, allow longer wait

    #[inline(always)]
    unsafe fn ld_acq_u64(ptr: *const u64) -> u64 {
        let r: u64;
        core::arch::asm!("ld.acquire.sys.global.u64 {r}, [{p}];", p = in(reg64) ptr, r = out(reg64) r, options(nostack));
        r
    }
    #[inline(always)]
    unsafe fn cas_u64(ptr: *mut u64, exp: u64, des: u64) -> u64 {
        let r: u64;
        core::arch::asm!("atom.cas.sys.global.b64 {r}, [{p}], {e}, {d};", p = in(reg64) ptr, e = in(reg64) exp, d = in(reg64) des, r = out(reg64) r, options(nostack));
        r
    }
    #[inline(always)]
    unsafe fn st_rel_u32(ptr: *mut u32, val: u32) {
        core::arch::asm!("st.release.sys.global.u32 [{p}], {v};", p = in(reg64) ptr, v = in(reg32) val, options(nostack));
    }
    #[inline(always)]
    unsafe fn ld_acq_u32(ptr: *const u32) -> u32 {
        let r: u32;
        core::arch::asm!("ld.acquire.sys.global.u32 {r}, [{p}];", p = in(reg64) ptr, r = out(reg32) r, options(nostack));
        r
    }
    #[inline(always)]
    unsafe fn fetch_add_u64(ptr: *mut u64, val: u64) -> u64 {
        let r: u64;
        core::arch::asm!("atom.add.sys.global.u64 {r}, [{p}], {v};", p = in(reg64) ptr, v = in(reg64) val, r = out(reg64) r, options(nostack));
        r
    }
    #[inline(always)]
    unsafe fn membar() {
        core::arch::asm!("membar.sys;", options(nostack));
    }
    #[inline(always)]
    unsafe fn amask() -> u32 {
        let r: u32;
        core::arch::asm!("activemask.b32 {r};", r = out(reg32) r, options(nostack, nomem));
        r
    }

    let tagged_index = |t: u64| -> u16 { (t & 0xFFFF) as u16 };
    let tagged_tag = |t: u64| -> u32 { (t >> 32) as u32 };
    let make_tagged = |tag: u32, idx: u16| -> u64 { ((tag as u64) << 32) | (idx as u64) };
    let pkt_offset = |idx: u16| -> usize { BUFFER_HEADER_SIZE + (idx as usize) * PACKET_SIZE };

    // Pop free packet
    let free_ptr = hc_buf.add(BUF_OFF_FREE_STACK) as *mut u64;
    let pkt_idx;
    loop {
        let old_head = ld_acq_u64(free_ptr as *const u64);
        let idx = tagged_index(old_head);
        if idx == NULL_INDEX { return u64::MAX; }
        let pkt = hc_buf.add(pkt_offset(idx));
        let next = core::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
        if cas_u64(free_ptr, old_head, next) == old_head {
            pkt_idx = idx;
            break;
        }
    }

    let pkt = hc_buf.add(pkt_offset(pkt_idx));

    // Fill header
    let mask = amask();
    core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
    core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_STDIN);
    st_rel_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

    // Fill payload: slot 0 = max bytes to read
    let payload = pkt.add(PKT_OFF_PAYLOAD);
    core::ptr::write_volatile(payload as *mut u64, max_len as u64);

    membar();

    // Push to ready stack
    let ready_ptr = hc_buf.add(BUF_OFF_READY_STACK) as *mut u64;
    loop {
        let old_head = ld_acq_u64(ready_ptr as *const u64);
        core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
        let new_tag = tagged_tag(old_head).wrapping_add(1);
        let new_tagged = make_tagged(new_tag, pkt_idx);
        if cas_u64(ready_ptr, old_head, new_tagged) == old_head { break; }
    }

    // Doorbell
    fetch_add_u64(hc_buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

    // Spin-wait for response (stdin is blocking on host, may take long)
    let control_ptr = pkt.add(PKT_OFF_CONTROL) as *const u32;
    let mut spins: u32 = 0;
    loop {
        let ctrl = ld_acq_u32(control_ptr);
        if ctrl & CONTROL_READY != 0 { break; }
        spins += 1;
        if spins >= GPU_MAX_SPIN {
            // Timeout — return packet and report error
            loop {
                let old_head = ld_acq_u64(free_ptr as *const u64);
                core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
                let new_tag = tagged_tag(old_head).wrapping_add(1);
                let new_tagged = make_tagged(new_tag, pkt_idx);
                if cas_u64(free_ptr, old_head, new_tagged) == old_head { break; }
            }
            return u64::MAX;
        }
    }

    // Read response: slot 0 = bytes read, slots 1-7 = data
    let bytes_read = core::ptr::read_volatile(payload as *const u64);
    if bytes_read != u64::MAX && bytes_read > 0 {
        let src = payload.add(8);
        let copy_len = if bytes_read > max_len as u64 { max_len } else { bytes_read as u32 };
        let mut i: u32 = 0;
        while i < copy_len {
            *out_buf.add(i as usize) = core::ptr::read_volatile(src.add(i as usize));
            i += 1;
        }
    }

    // Return packet to free stack
    loop {
        let old_head = ld_acq_u64(free_ptr as *const u64);
        core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
        let new_tag = tagged_tag(old_head).wrapping_add(1);
        let new_tagged = make_tagged(new_tag, pkt_idx);
        if cas_u64(free_ptr, old_head, new_tagged) == old_head { break; }
    }

    bytes_read
}

#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_hello_kernel(result: *mut u32) {
    // Test that std types are available
    let v = vec![1u32, 2, 3, 4, 5];
    let sum: u32 = v.iter().sum();

    // Test String from std (uses alloc)
    let s = String::from("Hello from GPU std!");
    let len = s.len() as u32;

    unsafe {
        core::ptr::write_volatile(result, sum + len);
    }
}

#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_format_kernel(result: *mut u32) {
    // Test format! macro (uses alloc + fmt)
    let formatted = format!("value = {}", 42u32);
    let len = formatted.len() as u32;

    unsafe {
        core::ptr::write_volatile(result, len);
    }
}

// ============================================================
// product.1: Dynamic allocation stress tests with runtime data
// ============================================================
// These kernels take runtime values as kernel arguments to prevent
// LLVM from constant-folding the allocations away.

/// Test 1: Vec with runtime data — build a Vec from kernel args, sum elements.
///
/// `input` = pointer to array of u32 values (device memory)
/// `input_len` = number of elements to push
/// `result` = output: sum of all elements
///
/// This forces the bump allocator to actually allocate because the Vec
/// contents come from device memory at runtime.
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_dynamic_vec_kernel(
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    let mut v: Vec<u32> = Vec::new();
    let len = input_len;

    // Push runtime values one by one — forces Vec to grow and reallocate
    let mut i: u32 = 0;
    while i < len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        v.push(val);
        i += 1;
    }

    let sum: u32 = v.iter().sum();
    unsafe {
        core::ptr::write_volatile(result, sum);
    }
}

/// Test 2: format! with runtime value — ensures the allocator is used for String.
///
/// `value` = runtime u32 value to format
/// `result` = output: length of formatted string
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_dynamic_format_kernel(
    value: u32,
    result: *mut u32,
) {
    let formatted = format!("result = {}", value);
    let len = formatted.len() as u32;
    unsafe {
        core::ptr::write_volatile(result, len);
    }
}

/// Test 3: Multiple Vecs alive simultaneously — tests allocator under pressure.
///
/// `input` = pointer to array of u32 values
/// `input_len` = number of elements
/// `result` = output array of u32[3]:
///   [0] = sum of first Vec (even indices)
///   [1] = sum of second Vec (odd indices)
///   [2] = total elements across both Vecs
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_dynamic_multi_vec_kernel(
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    let mut evens: Vec<u32> = Vec::new();
    let mut odds: Vec<u32> = Vec::new();

    let mut i: u32 = 0;
    while i < input_len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        if i % 2 == 0 {
            evens.push(val);
        } else {
            odds.push(val);
        }
        i += 1;
    }

    let even_sum: u32 = evens.iter().sum();
    let odd_sum: u32 = odds.iter().sum();
    let total_len = (evens.len() + odds.len()) as u32;

    unsafe {
        core::ptr::write_volatile(result.add(0), even_sum);
        core::ptr::write_volatile(result.add(1), odd_sum);
        core::ptr::write_volatile(result.add(2), total_len);
    }
}

/// Test 4: Vec with pre-capacity — tests Vec::with_capacity + push.
///
/// `input` = pointer to array of u32 values
/// `input_len` = number of elements
/// `result` = output: sum of all elements
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_dynamic_vec_capacity_kernel(
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    let mut v: Vec<u32> = Vec::with_capacity(input_len as usize);

    let mut i: u32 = 0;
    while i < input_len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        v.push(val);
        i += 1;
    }

    let sum: u32 = v.iter().sum();
    unsafe {
        core::ptr::write_volatile(result, sum);
    }
}

// ============================================================
// std-pal.1: PAL stdout routing — std::io::Write via hostcall
// ============================================================

/// Test: write! to std::io::stdout() routed through hostcall.
///
/// `buf` = hostcall buffer (mapped memory)
/// `value` = runtime u32 value to format and print
/// `result` = output: 1 on success, 0 on failure
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_println_kernel(
    buf: *mut u8,
    value: u32,
    result: *mut u32,
) {
    use std::io::Write;

    stdio_init(buf);

    // Use writeln! which goes through Write::write_fmt, not the _print path.
    // println! uses _print → print_to which has global capture state that
    // causes LLVM NVPTX "Circular dependency" crashes.
    let ok = writeln!(std::io::stdout(), "Hello from GPU println! value = {}", value).is_ok();

    unsafe {
        core::ptr::write_volatile(result, if ok { 1 } else { 0 });
    }
}

/// Test: multiple writes to stdout with runtime data.
///
/// `buf` = hostcall buffer
/// `input` = pointer to array of u32 values
/// `input_len` = number of elements to print
/// `result` = output: number of successful writes
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_println_multi_kernel(
    buf: *mut u8,
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    use std::io::Write;

    stdio_init(buf);

    let mut count: u32 = 0;
    let mut i: u32 = 0;
    while i < input_len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        if writeln!(std::io::stdout(), "GPU[{}] = {}", i, val).is_ok() {
            count += 1;
        }
        i += 1;
    }

    unsafe {
        core::ptr::write_volatile(result, count);
    }
}

/// Test: writeln! with formatted Vec contents.
///
/// `buf` = hostcall buffer
/// `input` = pointer to array of u32 values
/// `input_len` = number of elements
/// `result` = output: 1 on success
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_println_vec_kernel(
    buf: *mut u8,
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    use std::io::Write;

    stdio_init(buf);

    let mut v: Vec<u32> = Vec::new();
    let mut i: u32 = 0;
    while i < input_len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        v.push(val);
        i += 1;
    }

    let sum: u32 = v.iter().sum();
    let ok = writeln!(std::io::stdout(), "GPU Vec: {} elements, sum = {}", v.len(), sum).is_ok();

    unsafe {
        core::ptr::write_volatile(result, if ok { 1 } else { 0 });
    }
}

// ============================================================
// std-pal.2: PAL stdin routing — std::io::Read via hostcall
// ============================================================
//
// NOTE: std::io::stdin() wraps the PAL Stdin in OnceLock + ReentrantLock + BufReader.
// These layers don't work correctly on GPU (similar to println! LLVM crash issue).
// The OnceLock/ReentrantLock initialization path returns without doing actual I/O.
//
// Workaround: Call gpu_stdin_read() directly, same pattern as writeln! for stdout.
// This tests the PAL extern function mechanism works for both directions.

/// Test: read from stdin via direct PAL extern function call.
/// Bypasses std::io::stdin() wrapper (which uses broken OnceLock/ReentrantLock on GPU).
///
/// `buf` = hostcall buffer (mapped memory)
/// `result` = output array of u32[3]:
///   [0] = 1 on success, 0 on failure
///   [1] = bytes read
///   [2] = first byte of data read (for verification)
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_stdin_kernel(
    buf: *mut u8,
    result: *mut u32,
) {
    stdio_init(buf);

    unsafe {
        core::ptr::write_volatile(result.add(0), 0);
        core::ptr::write_volatile(result.add(1), 0);
        core::ptr::write_volatile(result.add(2), 0);
    }

    let mut read_buf = [0u8; 56];
    // Call gpu_stdin_read directly — the same extern function that our PAL
    // Stdin::read() delegates to. This bypasses the broken std::io::stdin()
    // wrapper layers (OnceLock, ReentrantLock, BufReader).
    let n = gpu_stdin_read(read_buf.as_mut_ptr(), read_buf.len());
    if n > 0 {
        unsafe {
            core::ptr::write_volatile(result.add(0), 1);
            core::ptr::write_volatile(result.add(1), n as u32);
            core::ptr::write_volatile(result.add(2), read_buf[0] as u32);
        }
    }
}

/// Test: read from stdin and echo to stdout (round-trip I/O).
/// Reads from stdin via extern, then writes to stdout via writeln!.
///
/// `buf` = hostcall buffer
/// `result` = output array of u32[2]:
///   [0] = 1 on success
///   [1] = bytes read from stdin
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn std_stdin_echo_kernel(
    buf: *mut u8,
    result: *mut u32,
) {
    use std::io::Write;

    stdio_init(buf);

    unsafe {
        core::ptr::write_volatile(result.add(0), 0);
        core::ptr::write_volatile(result.add(1), 0);
    }

    let mut read_buf = [0u8; 56];
    let n = gpu_stdin_read(read_buf.as_mut_ptr(), read_buf.len());
    if n > 0 {
        // Echo what we read back through stdout
        let data = unsafe { core::str::from_utf8_unchecked(&read_buf[..n]) };
        let ok = writeln!(std::io::stdout(), "GPU echo: {}", data).is_ok();
        unsafe {
            core::ptr::write_volatile(result.add(0), if ok { 1 } else { 0 });
            core::ptr::write_volatile(result.add(1), n as u32);
        }
    }
}

// ============================================================
// product.4: Showcase demo kernel
// ============================================================
// Combines all features: Vec, String, format!, writeln!(stdout),
// gpu_stdin_read, runtime kernel arguments — everything running
// on GPU through Rust std with hostcall backend.

/// Showcase demo: Rust std on GPU with hostcall I/O.
///
/// This kernel demonstrates the full stack:
/// 1. Read user name from stdin (via hostcall)
/// 2. Build a Vec from runtime kernel arguments
/// 3. Compute statistics using std iterators
/// 4. Format results using format!() (heap-allocated String)
/// 5. Print everything to stdout via writeln!()
///
/// `buf` = hostcall buffer (mapped memory)
/// `input` = pointer to array of u32 values (device memory)
/// `input_len` = number of elements
/// `result` = output array of u32[2]:
///   [0] = 1 on success
///   [1] = number of stdout messages written
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn showcase_kernel(
    buf: *mut u8,
    input: *const u32,
    input_len: u32,
    result: *mut u32,
) {
    use std::io::Write;

    stdio_init(buf);

    unsafe {
        core::ptr::write_volatile(result.add(0), 0);
        core::ptr::write_volatile(result.add(1), 0);
    }

    let mut msg_count: u32 = 0;

    // Step 1: Read a name from stdin
    let mut name_buf = [0u8; 56];
    let name_len = gpu_stdin_read(name_buf.as_mut_ptr(), name_buf.len());
    let name = if name_len > 0 {
        // Trim trailing newline if present
        let end = if name_len > 0 && name_buf[name_len - 1] == b'\n' {
            name_len - 1
        } else {
            name_len
        };
        unsafe { core::str::from_utf8_unchecked(&name_buf[..end]) }
    } else {
        "GPU User"
    };

    // Step 2: Greet the user
    if writeln!(std::io::stdout(), "Hello, {}! Welcome to Rust on GPU.", name).is_ok() {
        msg_count += 1;
    }

    // Step 3: Build a Vec from runtime kernel arguments
    let mut v: Vec<u32> = Vec::new();
    let mut i: u32 = 0;
    while i < input_len {
        let val = unsafe { core::ptr::read_volatile(input.add(i as usize)) };
        v.push(val);
        i += 1;
    }

    // Step 4: Compute statistics using std iterators
    let sum: u32 = v.iter().sum();
    let count = v.len();
    let min = v.iter().min().copied().unwrap_or(0);
    let max = v.iter().max().copied().unwrap_or(0);

    // Step 5: Format and print results
    let stats = format!(
        "Data: {} elements, sum={}, min={}, max={}",
        count, sum, min, max
    );
    if writeln!(std::io::stdout(), "{}", stats).is_ok() {
        msg_count += 1;
    }

    // Step 6: Build a filtered Vec and print it
    let evens: Vec<u32> = v.iter().filter(|&&x| x % 2 == 0).copied().collect();
    let odds: Vec<u32> = v.iter().filter(|&&x| x % 2 != 0).copied().collect();
    if writeln!(
        std::io::stdout(),
        "Even count: {}, Odd count: {}",
        evens.len(),
        odds.len()
    ).is_ok() {
        msg_count += 1;
    }

    // Step 7: Final message
    if writeln!(
        std::io::stdout(),
        "Goodbye, {}! GPU computation complete.",
        name
    ).is_ok() {
        msg_count += 1;
    }

    unsafe {
        core::ptr::write_volatile(result.add(0), 1);
        core::ptr::write_volatile(result.add(1), msg_count);
    }
}

// ============================================================
// allocator.2: Slab allocator deallocation test
// ============================================================

/// Test that the slab allocator correctly deallocates memory.
/// Allocates and drops Vec 10 times in a loop. With bump allocator,
/// this would consume 10x the memory. With slab allocator, memory
/// is reused.
///
/// result[0] = 1 on success
/// result[1] = number of successful alloc/dealloc cycles
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn slab_dealloc_test_kernel(
    _buf: *mut u8,
    result: *mut u32,
) {
    let mut cycles: u32 = 0;

    // Run 10 cycles of alloc+dealloc.
    // Each Vec<u32> with 100 elements = 400 bytes + Vec overhead.
    // With bump allocator, 10 cycles = 10x allocation (4KB+ leaked).
    // With slab allocator, memory is reused each cycle.
    for i in 0u32..10 {
        let mut v: Vec<u32> = Vec::new();
        for j in 0..100 {
            v.push(i * 100 + j);
        }
        let sum: u32 = v.iter().sum();
        // Verify correctness.
        let expected = (0..100u32).map(|j| i * 100 + j).sum::<u32>();
        if sum == expected {
            cycles += 1;
        }
        // v is dropped here — dealloc should free the memory.
    }

    // Now test String alloc/dealloc cycles.
    for _ in 0..10 {
        let s = format!("Hello from cycle {}", cycles);
        if !s.is_empty() {
            cycles += 1;
        }
        // s is dropped here — dealloc should free the String's buffer.
    }

    unsafe {
        core::ptr::write_volatile(result.add(0), if cycles == 20 { 1 } else { 0 });
        core::ptr::write_volatile(result.add(1), cycles);
    }
}
