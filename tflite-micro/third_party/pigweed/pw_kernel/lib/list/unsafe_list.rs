// Copyright 2025 The Pigweed Authors
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
#![allow(dead_code)]
use core::cell::UnsafeCell;
use core::cmp::Ordering;
use core::marker::PhantomData;
use core::ptr::NonNull;

// Intrusive link structures are particularly tricky in Rust because mutable
// references are expected to be globally unique.  Accessing the data through
// other methods is UB.  There is a good writeup of Tokio's soundness challenges
// at: https://gist.github.com/Darksonn/1567538f56af1a8038ecc3c664a42462.
//
// Here we adapt the strategy that Tokio uses:
// * The link pointer structure is marked with `PhantomPinned`.  This does two
//   things.  First it "poisons" the containing structure so that it's
//   unpinnable (immovable).  Second it triggers the heuristic from
//   https://github.com/rust-lang/rust/pull/82834 which will cause the compiler
//   to omit emitting a `noalias` annotation on mutable references to disable
//   compiler optimization that assume the mutable reference is unique.
// * `next` and `prev` themselves are accessed by `Link` through direct pointer
//   math on the `LinkInner` struct.  This avoids creating mutable references to
//   those fields themselves which can not be annotated with `PhantomPinned`.
//   `LinkInner` is `#[repr(C)]` to make the pointer math deterministic.
//   `LinkInner` is declared in an `inner` module so that `next` and `prev` can
//   not be accessed directly be the rest of the code.
//
// TODO: konkers - Understand if we need to annotate the alignment of LinkInner.
mod inner {
    use core::marker::PhantomPinned;
    use core::mem::offset_of;
    use core::ptr::NonNull;

    use super::Link;

    #[repr(C)]
    pub struct LinkInner {
        #[allow(dead_code)]
        next: Option<NonNull<Link>>,
        #[allow(dead_code)]
        prev: Option<NonNull<Link>>,
        _pin: PhantomPinned,
    }

    impl LinkInner {
        pub const NEXT_OFFSET: usize = offset_of!(LinkInner, next);
        pub const PREV_OFFSET: usize = offset_of!(LinkInner, prev);
        pub const UNLINKED_VALUE: Option<NonNull<Link>> =
            Some(unsafe { NonNull::new_unchecked(usize::MAX as *mut Link) });

        pub const fn new() -> Self {
            Self {
                next: Self::UNLINKED_VALUE,
                prev: Self::UNLINKED_VALUE,
                _pin: PhantomPinned,
            }
        }
    }
}
use inner::LinkInner;

pub struct Link {
    // UnsafeCell here is used to allow the code to access the data mutably.
    // Bare mutable pointer access is unsound without this.
    inner: UnsafeCell<LinkInner>,
}

// SAFETY:
//
// When used with a ForeignList (or RandomAccessForeignList), a given node can
// only be in a single list at a time.  This list logically owns the node for
// the duration of its membership in the list.  All mutation of the node's
// `Link` pointers are done while the node is in the list.  There is no API to
// get a mutable reference to a node while it is in the list nor directly
// manipulate its membership or position in the list.
//
// When used with an `UnsafeList` it is the users responsibility to ensure that
// Rust's read-shared/write-exclusive semantics are upheld across all access
// to a given node including when that node is also in a ForeignList
// RandomAccessForeignList.
unsafe impl Send for Link {}
unsafe impl Sync for Link {}

#[inline]
unsafe fn get_element(inner: &UnsafeCell<LinkInner>, offset: usize) -> Option<NonNull<Link>> {
    let inner_ptr = inner.get().cast::<Option<NonNull<Link>>>();
    unsafe {
        let element_ptr = inner_ptr.byte_add(offset);
        core::ptr::read(element_ptr)
    }
}

#[inline]
unsafe fn set_element(inner: &UnsafeCell<LinkInner>, offset: usize, value: Option<NonNull<Link>>) {
    let inner_ptr = inner.get().cast::<Option<NonNull<Link>>>();
    unsafe {
        let element_ptr = inner_ptr.byte_add(offset);
        core::ptr::write(element_ptr, value);
    }
}

impl Link {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: UnsafeCell::new(LinkInner::new()),
        }
    }

    #[must_use]
    pub fn is_unlinked(&self) -> bool {
        self.get_next() == LinkInner::UNLINKED_VALUE && self.get_prev() == LinkInner::UNLINKED_VALUE
    }

    #[must_use]
    pub fn is_linked(&self) -> bool {
        !self.is_unlinked()
    }

    fn set_unlinked(&mut self) {
        self.set_next(LinkInner::UNLINKED_VALUE);
        self.set_prev(LinkInner::UNLINKED_VALUE);
    }

    #[inline]
    #[must_use]
    fn get_next(&self) -> Option<NonNull<Link>> {
        unsafe { get_element(&self.inner, LinkInner::NEXT_OFFSET) }
    }

    #[inline]
    fn set_next(&mut self, value: Option<NonNull<Link>>) {
        unsafe { set_element(&self.inner, LinkInner::NEXT_OFFSET, value) }
    }

    #[inline]
    #[must_use]
    fn get_prev(&self) -> Option<NonNull<Link>> {
        unsafe { get_element(&self.inner, LinkInner::PREV_OFFSET) }
    }

    #[inline]
    fn set_prev(&mut self, value: Option<NonNull<Link>>) {
        unsafe { set_element(&self.inner, LinkInner::PREV_OFFSET, value) }
    }
}

impl Default for Link {
    fn default() -> Self {
        Self::new()
    }
}

// `UnsafeList` uses None to mark the beginning and end of lists as opposed to
// pointers to the base list node.  This means that there are never pointers to
// `UnsafeList` and the same care is not needed to avoid mutable references as
// is taken with the `Link` structure.
pub struct UnsafeList<T, A: Adapter> {
    head: Option<NonNull<Link>>,
    tail: Option<NonNull<Link>>,
    _phantom_type: PhantomData<T>,
    _phantom_adapter: PhantomData<A>,
}

pub trait Adapter {
    const LINK_OFFSET: usize;
}

/// Defines an adapter type and implements [`Adapter`] for it.
///
/// # Usage
///
/// `define_adapter!` accepts a syntax like the following (where type arguments
/// are optional):
///
/// ```
/// use list::{Link, define_adapter};
///
/// struct Node<T: Copy> {
///     data: T,
///     link: Link,
/// }
/// define_adapter!(pub Adapter<T: Copy> => Node<T>::link);
/// ```
#[macro_export]
macro_rules! define_adapter {
    ($vis:vis $name:ident $(<$($tyvar:ident $(: $bound:path)?),*>)? => $node:ident $(<$($node_tyvar:ident),*>)? :: $link:ident) => {
        $vis struct $name $(<$($tyvar $(: $bound)?),*>)? {
            $(
                _marker: core::marker::PhantomData<$($tyvar),*>,
            )?
        }

        impl $(<$($tyvar $(: $bound)?),*>)? $crate::Adapter for $name $(<$($tyvar),*>)? {
            const LINK_OFFSET: usize = core::mem::offset_of!($node $(<$($node_tyvar),*>)?, $link);
        }
    };
}

impl<T, A: Adapter> UnsafeList<T, A> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            head: None,
            tail: None,
            _phantom_type: PhantomData,
            _phantom_adapter: PhantomData,
        }
    }

    /// # Safety
    /// It is up to the caller to ensure exclusive access to the list and its
    /// members.
    #[must_use]
    pub unsafe fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    unsafe fn get_link_ptr(element: NonNull<T>) -> NonNull<Link> {
        unsafe {
            let element_ptr: NonNull<Link> =
                core::mem::transmute::<NonNull<T>, NonNull<Link>>(element);
            element_ptr.byte_add(A::LINK_OFFSET)
        }
    }

    unsafe fn get_element_ptr(link: NonNull<Link>) -> *const T {
        unsafe { link.byte_sub(A::LINK_OFFSET).as_ptr().cast::<T>() }
    }

    unsafe fn get_element_mut(link: NonNull<Link>) -> *mut T {
        unsafe { link.byte_sub(A::LINK_OFFSET).as_ptr().cast::<T>() }
    }

    /// Returns true if element is in **ANY** list that uses this list's adapter.
    ///
    /// # Safety
    /// `element` must be a valid, non-null pointer.
    pub unsafe fn is_element_linked(&mut self, element: *mut T) -> bool {
        unsafe {
            let element = NonNull::new_unchecked(element);
            let link_ptr = Self::get_link_ptr(element);
            (*link_ptr.as_ptr()).is_linked()
        }
    }

    /// unchecked means:
    /// * we don't `assert!((*element_ptr.as_ptr()).is_unlinked());`
    /// * we don't check that `element` is non-null.
    ///
    /// # Safety
    /// It is up to the caller to ensure exclusive access to the list and its
    /// members.
    /// It is up to the caller to ensure the element is not in a list.
    /// It is up to the caller to ensure the element is non-null.
    pub unsafe fn push_front_unchecked(&mut self, element: *mut T) {
        let link_ptr = unsafe {
            let element = NonNull::new_unchecked(element);
            let link_ptr = Self::get_link_ptr(element);

            // Link up the added element.
            (*link_ptr.as_ptr()).set_next(self.head);
            (*link_ptr.as_ptr()).set_prev(None);

            match self.head {
                // If `head` was `None`, the list is empty and `tail` should point
                // to the added element.
                None => self.tail = Some(link_ptr),

                // If `head` is not `None`, point the previous `head` to the added
                // element.
                Some(head) => (*head.as_ptr()).set_prev(Some(link_ptr)),
            }

            link_ptr
        };

        // Finally point `head` to the added element.
        self.head = Some(link_ptr);
    }

    /// unchecked means:
    /// * we don't `assert!((*element_ptr.as_ptr()).is_unlinked());`
    /// * we don't check that `element` is non-null.
    ///
    /// # Safety
    /// It is up to the caller to ensure exclusive access to the list and its
    /// members.
    /// It is up to the caller to ensure the element is not in a list.
    /// It is up to the caller to ensure the element is non-null.
    pub unsafe fn push_back_unchecked(&mut self, element: *mut T) {
        let link_ptr = unsafe {
            let element = NonNull::new_unchecked(element);
            let link_ptr = Self::get_link_ptr(element);

            // Link up the added element.
            (*link_ptr.as_ptr()).set_next(None);
            (*link_ptr.as_ptr()).set_prev(self.tail);

            match self.tail {
                // If `tail` was `None`, the list is empty and `head` should point
                // to the added element.
                None => self.head = Some(link_ptr),

                // If `tail` is not `None`, point the previous `tail` to the added
                // element.
                Some(tail) => (*tail.as_ptr()).set_next(Some(link_ptr)),
            }

            link_ptr
        };

        // Finally point `tail` to the added element.
        self.tail = Some(link_ptr);
    }

    /// Insert element_a into the list immediately before `element_b`
    ///
    /// # Safety
    /// It is up to the caller to ensure exclusive access to the list and its
    /// members.
    /// It is up to the caller to ensure that element_a is not in a list.
    /// It is up to the caller to ensure that element_b is in a list.
    /// It is up to the caller to ensure the element_a and element_b are non-null.
    unsafe fn insert_before(&mut self, element_a: NonNull<Link>, element_b: NonNull<Link>) {
        unsafe {
            let prev = (*element_b.as_ptr()).get_prev();

            (*element_a.as_ptr()).set_next(Some(element_b));
            (*element_a.as_ptr()).set_prev(prev);

            (*element_b.as_ptr()).set_prev(Some(element_a));

            match prev {
                // Element is at the head of the list
                None => self.head = Some(element_a),

                // Element has elements before it in the list.
                Some(prev_ptr) => (*prev_ptr.as_ptr()).set_next(Some(element_a)),
            }
        }
    }

    /// unlinks element from the linked list.
    ///
    /// # Safety
    /// It is up to the caller to ensure exclusive access to the list and its
    /// members.
    /// It is up to the caller to ensure the element is in the list
    /// It is up to the caller to ensure the element is non-null.
    pub unsafe fn unlink_element_unchecked(&mut self, element: *mut T) {
        let (link_ptr, prev, next) = unsafe {
            let element = NonNull::new_unchecked(element);
            let link_ptr = Self::get_link_ptr(element);
            (
                link_ptr,
                (*link_ptr.as_ptr()).get_prev(),
                (*link_ptr.as_ptr()).get_next(),
            )
        };

        match prev {
            // Element is at the head of the list
            None => self.head = next,

            // Element has elements before it in the list.
            Some(prev_ptr) => unsafe { (*prev_ptr.as_ptr()).set_next(next) },
        }

        match next {
            // Element is at the tail of the list
            None => self.tail = prev,

            // Element has elements after it in the list.
            Some(next_ptr) => unsafe { (*next_ptr.as_ptr()).set_prev(prev) },
        }

        unsafe {
            (*link_ptr.as_ptr()).set_unlinked();
        }
    }

    /// # Safety
    /// Call ensures the element is a valid point to an instance of T.
    pub unsafe fn unlink_element(&mut self, element: NonNull<T>) -> Option<NonNull<T>> {
        unsafe {
            if (*Self::get_link_ptr(element).as_ptr()).is_linked() {
                self.unlink_element_unchecked(element.as_ptr());
                Some(element)
            } else {
                None
            }
        }
    }

    /// # Safety
    /// It is up to the caller to ensure exclusive access to the list and its
    /// members.
    pub unsafe fn for_each<E, F: FnMut(&T) -> Result<(), E>>(
        &self,
        mut callback: F,
    ) -> Result<(), E> {
        let mut cur = self.head;

        loop {
            let Some(cur_ptr) = cur else {
                break;
            };

            unsafe {
                let element = Self::get_element_ptr(cur_ptr);
                callback(&*element)?;

                cur = (*cur_ptr.as_ptr()).get_next();
            }
        }

        Ok(())
    }

    /// Filter iterates over every element in the list calling `callback` on
    /// each one.  If `callback` returns false, the element will be removed
    /// from the list without modifying the element itself.  It is safe to
    /// add the element to the another linked list within `callback` if it
    /// returns false.
    ///
    /// # Safety
    /// It is up to the caller to ensure exclusive access to the list and its
    /// members.
    pub unsafe fn filter<F: FnMut(&mut T) -> bool>(&mut self, mut callback: F) {
        let mut cur = self.head;

        loop {
            let Some(cur_ptr) = cur else {
                break;
            };

            let next = unsafe {
                let element = Self::get_element_mut(cur_ptr);

                // Cache the next element so that we don't rely on `element` staying
                // coherent across calls to `callback`.
                let next = (*cur_ptr.as_ptr()).get_next();

                if !callback(&mut *element) {
                    self.unlink_element_unchecked(element);
                }
                next
            };

            cur = next;
        }
    }

    /// Return a reference to the first element in the list, clearing the
    /// prev/next fields in the element.
    ///
    /// TODO: reevalulate the lifetime marker here, since it is a lie.
    ///
    /// # Safety
    /// It is up to the caller to ensure exclusive access to the list and its
    /// members.
    pub unsafe fn pop_head(&mut self) -> Option<*mut T> {
        let cur = self.head?;

        unsafe {
            let element = Self::get_element_mut(cur);

            self.unlink_element_unchecked(element);
            Some(element)
        }
    }
}

impl<T, A: Adapter> UnsafeList<T, A> {
    /// unchecked means:
    /// * we don't `assert!((*element_ptr.as_ptr()).is_unlinked());`
    /// * we don't check that `element` is non-null.
    ///
    /// # Safety
    /// It is up to the caller to ensure exclusive access to the list and its
    /// members.
    /// It is up to the caller to ensure the element is not in a list.
    /// It is up to the caller to ensure the element is non-null.
    pub unsafe fn sorted_insert_by_unchecked<F: FnMut(&T, &T) -> Ordering>(
        &mut self,
        element: *mut T,
        mut compare: F,
    ) {
        let element = unsafe { NonNull::new_unchecked(element) };
        let link_ptr = unsafe { Self::get_link_ptr(element) };

        let mut cur = self.head;

        loop {
            let Some(cur_link_ptr) = cur else {
                break;
            };

            unsafe {
                let cur_element_ptr = Self::get_element_ptr(cur_link_ptr);
                if let Ordering::Less | Ordering::Equal =
                    compare(&*element.as_ptr(), &*cur_element_ptr)
                {
                    self.insert_before(link_ptr, cur_link_ptr);
                    return;
                }
            }

            cur = unsafe { (*cur_link_ptr.as_ptr()).get_next() };
        }

        // Either the list is empty or all elements are less than `element`.
        // In both cases pushing the element to the back is appropriate.
        unsafe {
            self.push_back_unchecked(element.as_ptr());
        }
    }
}

impl<T, A: Adapter> Default for UnsafeList<T, A> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use core::ptr::NonNull;

    use unittest::test;

    use super::*;

    // `#[repr(C)]` is used to ensure that `link` is at a non-zero offset.
    // Previously, without this, the compiler was putting link at the beginning of
    // the struct causing `LINK_OFFSET` in the adapter to be zero which obfuscated
    // some pointer math bugs.
    #[repr(C)]
    struct TestMember {
        value: u32,
        link: Link,
    }

    impl PartialEq for TestMember {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Eq for TestMember {}

    impl PartialOrd for TestMember {
        fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
            Some(self.value.cmp(&other.value))
        }
    }

    impl Ord for TestMember {
        fn cmp(&self, other: &Self) -> core::cmp::Ordering {
            self.value.cmp(&other.value)
        }
    }

    define_adapter!(TestAdapter => TestMember::link);

    unsafe fn validate_list(
        list: &UnsafeList<TestMember, TestAdapter>,
        expected_values: &[u32],
    ) -> unittest::Result<()> {
        let mut index = 0;
        unsafe {
            list.for_each(|element| {
                unittest::assert_eq!(element.value, expected_values[index]);
                index += 1;
                Ok(())
            })
        }?;

        unittest::assert_eq!(index, expected_values.len());
        Ok(())
    }

    #[test]
    fn new_link_is_not_linked() -> unittest::Result<()> {
        let link = Link::new();
        unittest::assert_false!(link.is_linked());
        unittest::assert_true!(link.is_unlinked());
        Ok(())
    }

    #[test]
    fn new_list_is_empty() -> unittest::Result<()> {
        let list = UnsafeList::<TestMember, TestAdapter>::new();
        unittest::assert_true!(unsafe { list.is_empty() });
        Ok(())
    }

    #[test]
    fn single_element_list_is_non_empty_and_linked() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut list = UnsafeList::<TestMember, TestAdapter>::new();

        unittest::assert_true!(unsafe { list.is_empty() });
        unittest::assert_false!(unsafe { list.is_element_linked(&raw mut element1) });

        unsafe { list.push_front_unchecked(&raw mut element1) };

        unittest::assert_false!(unsafe { list.is_empty() });
        unittest::assert_true!(unsafe { list.is_element_linked(&raw mut element1) });

        Ok(())
    }

    #[test]
    fn push_front_adds_in_correct_order() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.push_front_unchecked(&mut element2) };
        unsafe { list.push_front_unchecked(&mut element1) };

        unittest::assert_false!(unsafe { list.is_empty() });

        unsafe { validate_list(&list, &[1, 2]) }
    }

    #[test]
    fn push_back_adds_in_correct_order() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.push_back_unchecked(&mut element2) };
        unsafe { list.push_back_unchecked(&mut element1) };

        unittest::assert_false!(unsafe { list.is_empty() });

        unsafe { validate_list(&list, &[2, 1]) }
    }

    #[test]
    fn unlink_removes_head_correctly() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.push_front_unchecked(&mut element3) };
        unsafe { list.push_front_unchecked(&mut element2) };
        unsafe { list.push_front_unchecked(&mut element1) };

        unsafe { list.unlink_element_unchecked(&mut element1) };

        unsafe { validate_list(&list, &[2, 3]) }
    }

    #[test]
    fn unlink_removes_tail_correctly() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.push_front_unchecked(&mut element3) };
        unsafe { list.push_front_unchecked(&mut element2) };
        unsafe { list.push_front_unchecked(&mut element1) };

        unsafe { list.unlink_element_unchecked(&mut element3) };

        unsafe { validate_list(&list, &[1, 2]) }
    }

    #[test]
    fn unlink_removes_middle_correctly() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.push_front_unchecked(&mut element3) };
        unsafe { list.push_front_unchecked(&mut element2) };
        unsafe { list.push_front_unchecked(&mut element1) };

        unsafe { list.unlink_element_unchecked(&mut element2) };

        unsafe { validate_list(&list, &[1, 3]) }
    }

    #[test]
    fn unlink_fails_non_inserted_element() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();

        unittest::assert_eq!(
            unsafe { list.unlink_element(NonNull::new(&raw mut element1).unwrap()) },
            None
        );
        Ok(())
    }

    #[test]
    fn pop_head_removes_correctly() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.push_front_unchecked(&mut element1) };
        unsafe { list.push_front_unchecked(&mut element2) };
        unsafe { list.push_front_unchecked(&mut element3) };

        unsafe {
            let e = list.pop_head();
            unittest::assert_true!(e.is_some());
            let e = e.unwrap();
            unittest::assert_eq!((*e).value, 3);
            unittest::assert_true!((*e).link.is_unlinked());
        }

        unsafe {
            let e = list.pop_head();
            unittest::assert_true!(e.is_some());
            let e = e.unwrap();
            unittest::assert_eq!((*e).value, 2);
            unittest::assert_true!((*e).link.is_unlinked());
        }

        unsafe {
            let e = list.pop_head();
            unittest::assert_true!(e.is_some());
            let e = e.unwrap();
            unittest::assert_eq!((*e).value, 1);
            unittest::assert_true!((*e).link.is_unlinked());
        }

        unsafe { validate_list(&list, &[]) }
    }

    #[test]
    fn filter_removes_nothing_correctly() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.push_front_unchecked(&mut element3) };
        unsafe { list.push_front_unchecked(&mut element2) };
        unsafe { list.push_front_unchecked(&mut element1) };

        unsafe { list.filter(|_| true) };

        unsafe { validate_list(&list, &[1, 2, 3]) }
    }

    #[test]
    fn filter_removes_everything_correctly() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.push_front_unchecked(&mut element3) };
        unsafe { list.push_front_unchecked(&mut element2) };
        unsafe { list.push_front_unchecked(&mut element1) };

        unsafe { list.filter(|_| false) };

        unsafe { validate_list(&list, &[]) }
    }

    #[test]
    fn filter_removes_head_correctly() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.push_front_unchecked(&mut element3) };
        unsafe { list.push_front_unchecked(&mut element2) };
        unsafe { list.push_front_unchecked(&mut element1) };

        unsafe { list.filter(|element| element.value != 1) };

        unsafe { validate_list(&list, &[2, 3]) }
    }

    #[test]
    fn filter_removes_middle_correctly() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.push_front_unchecked(&mut element3) };
        unsafe { list.push_front_unchecked(&mut element2) };
        unsafe { list.push_front_unchecked(&mut element1) };

        unsafe { list.filter(|element| element.value != 2) };

        unsafe { validate_list(&list, &[1, 3]) }
    }

    #[test]
    fn filter_removes_tail_correctly() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.push_front_unchecked(&mut element3) };
        unsafe { list.push_front_unchecked(&mut element2) };
        unsafe { list.push_front_unchecked(&mut element1) };

        unsafe { list.filter(|element| element.value != 3) };

        unsafe { validate_list(&list, &[1, 2]) }
    }

    #[test]
    fn sorted_insert_inserts_sorted_items_in_correct_order() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.sorted_insert_by_unchecked(&mut element3, TestMember::cmp) };
        unsafe { list.sorted_insert_by_unchecked(&mut element2, TestMember::cmp) };
        unsafe { list.sorted_insert_by_unchecked(&mut element1, TestMember::cmp) };
        unsafe { validate_list(&list, &[1, 2, 3]) }
    }

    #[test]
    fn sorted_insert_inserts_reverse_sorted_items_in_correct_order() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.sorted_insert_by_unchecked(&mut element1, TestMember::cmp) };
        unsafe { list.sorted_insert_by_unchecked(&mut element2, TestMember::cmp) };
        unsafe { list.sorted_insert_by_unchecked(&mut element3, TestMember::cmp) };
        unsafe { validate_list(&list, &[1, 2, 3]) }
    }

    #[test]
    fn sorted_insert_inserts_unsorted_items_in_correct_order() -> unittest::Result<()> {
        let mut element1 = TestMember {
            value: 1,
            link: Link::new(),
        };
        let mut element2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element2_2 = TestMember {
            value: 2,
            link: Link::new(),
        };
        let mut element3 = TestMember {
            value: 3,
            link: Link::new(),
        };

        let mut list = UnsafeList::<TestMember, TestAdapter>::new();
        unsafe { list.sorted_insert_by_unchecked(&mut element2, TestMember::cmp) };
        unsafe { list.sorted_insert_by_unchecked(&mut element1, TestMember::cmp) };
        unsafe { list.sorted_insert_by_unchecked(&mut element3, TestMember::cmp) };
        unsafe { list.sorted_insert_by_unchecked(&mut element2_2, TestMember::cmp) };
        unsafe { validate_list(&list, &[1, 2, 2, 3]) }
    }
}
