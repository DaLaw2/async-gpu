#![no_std]
#![no_main]
#![feature(register_attr)]
#![register_attr(warp_cooperative)]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[warp_cooperative]
async fn cooperative_poll() -> u32 {
    42
}

#[no_mangle]
pub extern "C" fn kernel_main() {
    let _future = cooperative_poll();
}
