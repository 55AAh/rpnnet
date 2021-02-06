#[macro_export]
macro_rules! MakeConsumable {
    ($name:ident,$inner_type:tt,$obj_name:ident) => {
        pub(super) struct $name {
            obj: Option<$inner_type>,
        }
        impl $name {
            pub(super) fn acquire(val: $inner_type) -> Self {
                Self { obj: Some(val) }
            }
            fn get_ref(self: &Self) -> &$inner_type {
                self.obj
                    .as_ref()
                    .expect(std::stringify!(This $obj_name object is consumed and cannot be used))
            }
            fn get_ref_mut(self: &mut Self) -> &mut $inner_type {
                self.obj
                    .as_mut()
                    .expect(std::stringify!(This $obj_name object is consumed and cannot be used))
            }
            fn release(&mut self) -> $inner_type {
                self.obj
                    .take()
                    .expect(std::stringify!(This $obj_name object is consumed and cannot be used))
            }
        }
    };
}

#[macro_export]
macro_rules! Impl_to_PyErr {
    (for $($t:ty),+) => {
        $(impl From<$t> for PyErr {
            fn from(err: $t) -> Self {
                pyo3::exceptions::PyValueError::new_err(format!("{}", err))
            }
        }
        )*
    }
}
