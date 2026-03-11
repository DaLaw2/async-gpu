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
