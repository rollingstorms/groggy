//! Datetime translation only; canonicalization and validation live in Rust types.
use super::super::helpers::graph_error_to_py_err;
use groggy::{AttrValue, CalendarDate, FixedDuration, UtcInstant};
use pyo3::types::{PyDate, PyDateAccess, PyDateTime, PyDelta, PyDeltaAccess, PyTimeAccess};
use pyo3::{exceptions::PyTypeError, prelude::*};

pub fn from_python(value: &PyAny) -> PyResult<Option<AttrValue>> {
    if value.is_instance_of::<PyDateTime>() {
        if value.call_method0("utcoffset")?.is_none() {
            return Err(PyTypeError::new_err(
                "UTC instant requires an aware datetime; naive datetime is unsupported",
            ));
        }
        let utc = value
            .py()
            .import("datetime")?
            .getattr("timezone")?
            .getattr("utc")?;
        let normalized = value.call_method1("astimezone", (utc,))?;
        let date = normalized.downcast::<PyDateTime>()?;
        let calendar = CalendarDate::new(
            date.get_year(),
            date.get_month().into(),
            date.get_day().into(),
        )
        .map_err(graph_error_to_py_err)?;
        return UtcInstant::from_calendar(
            calendar,
            date.get_hour().into(),
            date.get_minute().into(),
            date.get_second().into(),
            date.get_microsecond(),
        )
        .map(AttrValue::UtcInstant)
        .map(Some)
        .map_err(graph_error_to_py_err);
    }
    if let Ok(date) = value.downcast::<PyDate>() {
        return CalendarDate::new(
            date.get_year(),
            date.get_month().into(),
            date.get_day().into(),
        )
        .map(AttrValue::Date)
        .map(Some)
        .map_err(graph_error_to_py_err);
    }
    if let Ok(delta) = value.downcast::<PyDelta>() {
        return FixedDuration::new(
            delta.get_days(),
            delta.get_seconds() as u32,
            delta.get_microseconds() as u32,
        )
        .map(AttrValue::Duration)
        .map(Some)
        .map_err(graph_error_to_py_err);
    }
    Ok(None)
}

pub fn instant_to_python(py: Python, value: UtcInstant) -> PyResult<PyObject> {
    let (date, hour, minute, second) = value.calendar().map_err(graph_error_to_py_err)?;
    let utc = py.import("datetime")?.getattr("timezone")?.getattr("utc")?;
    Ok(PyDateTime::new(
        py,
        date.year(),
        date.month() as u8,
        date.day() as u8,
        hour as u8,
        minute as u8,
        second as u8,
        value.microseconds(),
        Some(utc.downcast()?),
    )?
    .to_object(py))
}
pub fn date_to_python(py: Python, value: CalendarDate) -> PyResult<PyObject> {
    Ok(PyDate::new(py, value.year(), value.month() as u8, value.day() as u8)?.to_object(py))
}
pub fn duration_to_python(py: Python, value: FixedDuration) -> PyResult<PyObject> {
    Ok(PyDelta::new(
        py,
        value.days(),
        value.seconds() as i32,
        value.microseconds() as i32,
        false,
    )?
    .to_object(py))
}
