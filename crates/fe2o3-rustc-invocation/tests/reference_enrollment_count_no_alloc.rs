//! This separate test binary observes allocations in the projection, including errors.
use fe2o3_rustc_invocation::ReferenceEnrollmentRequestV1 as Request;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::convert::Infallible;

struct ObservedAllocator;
thread_local! {
    static CALLS: Cell<Option<usize>> = const { Cell::new(None) };
}
fn observe() {
    let _ = CALLS.try_with(|calls| {
        if let Some(n) = calls.get() {
            calls.set(Some(n + 1));
        }
    });
}
unsafe impl GlobalAlloc for ObservedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        observe();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        observe();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        observe();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: ObservedAllocator = ObservedAllocator;

fn measured<T>(run: impl FnOnce() -> T) -> (T, usize) {
    CALLS.with(|calls| {
        assert!(calls.get().is_none());
        calls.set(Some(0));
    });
    let value = run();
    let count = CALLS.with(|calls| calls.replace(None).unwrap());
    (value, count)
}

#[test]
fn count_projection_has_no_heap_activity_on_success_or_refusal() {
    let valid = r#"{"version":1,"bindings":[{"kernel":"a\u0062","reference":"r\ud83d\ude80"}]}"#;
    // Positive control: the allocator observer must see the materializing decoder.
    let (full, allocated) = measured(|| Request::decode(valid, |_| Ok::<_, Infallible>(())));
    assert!(full.is_ok());
    assert!(allocated > 0);
    let long = format!("{valid}{}", " ".repeat(4096 - valid.len()));
    let excessive = format!("{long} ");
    let corpus = [
        valid,
        long.as_str(),
        excessive.as_str(),
        "",
        "[",
        "{}",
        r#"{"version":1,"bindings":[]}"#,
        r#"{"version":1,"bindings":[{"kernel":"","reference":"r"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"\ud800","reference":"r"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"b","reference":"r"},{"kernel":"a","reference":"r"}]}"#,
        r#"{"vers\u0069on":1,"version":1,"bindings":[]}"#,
    ];
    for text in corpus {
        let (projected, allocations) =
            measured(|| Request::project_binding_count(text, |_| Ok::<_, Infallible>(())));
        assert_eq!(allocations, 0, "{text:?}");
        let full = Request::decode(text, |_| Ok::<_, Infallible>(())).map(|r| r.bindings().len());
        assert_eq!(projected, full);
    }
    for stop in 1..=2 {
        let mut calls = 0;
        let (result, allocations) = measured(|| {
            Request::project_binding_count("not-json", |_| {
                calls += 1;
                if calls == stop { Err(stop) } else { Ok(()) }
            })
        });
        assert!(result.is_err());
        assert_eq!(allocations, 0);
        assert_eq!(calls, stop);
    }
}
