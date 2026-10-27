//! Python datetime's C API is unavailable through the stable limited ABI.
use super::super::helpers::graph_error_to_py_err;
use groggy::{AttrValue, CalendarDate, FixedDuration, UtcInstant};
use pyo3::{exceptions::PyTypeError, once_cell::GILOnceCell, prelude::*, types::PyType};

struct TemporalTypes {
    datetime: Py<PyType>,
    date: Py<PyType>,
    duration: Py<PyType>,
    utc: PyObject,
}
static TYPES: GILOnceCell<TemporalTypes> = GILOnceCell::new();

fn types(py: Python<'_>) -> PyResult<&TemporalTypes> {
    TYPES.get_or_try_init(py, || {
        let module = py.import("datetime")?;
        Ok(TemporalTypes {
            datetime: module.getattr("datetime")?.extract()?,
            date: module.getattr("date")?.extract()?,
            duration: module.getattr("timedelta")?.extract()?,
            utc: module.getattr("timezone")?.getattr("utc")?.into(),
        })
    })
}

// Read the built-in descriptor, like the C API. Subclass properties must not
// replace stored calendar fields, nor can an unrelated __class__ spoof opt in.
fn field<'py, T: FromPyObject<'py>>(
    class: &'py PyType,
    value: &'py PyAny,
    name: &str,
) -> PyResult<T> {
    class
        .getattr(name)?
        .call_method1("__get__", (value,))?
        .extract()
}

fn calendar(class: &PyType, value: &PyAny) -> PyResult<CalendarDate> {
    CalendarDate::new(
        field(class, value, "year")?,
        field(class, value, "month")?,
        field(class, value, "day")?,
    )
    .map_err(graph_error_to_py_err)
}

pub fn from_python(value: &PyAny) -> PyResult<Option<AttrValue>> {
    let py = value.py();
    let types = types(py)?;
    let datetime = types.datetime.as_ref(py);
    let date = types.date.as_ref(py);
    let duration = types.duration.as_ref(py);
    let actual = value.get_type();
    if actual.is_subclass(datetime)? {
        if value.call_method0("utcoffset")?.is_none() {
            return Err(PyTypeError::new_err(
                "UTC instant requires an aware datetime; naive datetime is unsupported",
            ));
        }
        let normalized = value.call_method1("astimezone", (types.utc.as_ref(py),))?;
        if !normalized.get_type().is_subclass(datetime)? {
            return Err(PyTypeError::new_err("astimezone must return a datetime"));
        }
        return UtcInstant::from_calendar(
            calendar(datetime, normalized)?,
            field(datetime, normalized, "hour")?,
            field(datetime, normalized, "minute")?,
            field(datetime, normalized, "second")?,
            field(datetime, normalized, "microsecond")?,
        )
        .map(AttrValue::UtcInstant)
        .map(Some)
        .map_err(graph_error_to_py_err);
    }
    if actual.is_subclass(date)? {
        return calendar(date, value).map(AttrValue::Date).map(Some);
    }
    if actual.is_subclass(duration)? {
        return FixedDuration::new(
            field(duration, value, "days")?,
            field(duration, value, "seconds")?,
            field(duration, value, "microseconds")?,
        )
        .map(AttrValue::Duration)
        .map(Some)
        .map_err(graph_error_to_py_err);
    }
    Ok(None)
}

pub fn instant_to_python(py: Python<'_>, value: UtcInstant) -> PyResult<PyObject> {
    let (date, hour, minute, second) = value.calendar().map_err(graph_error_to_py_err)?;
    let types = types(py)?;
    Ok(types
        .datetime
        .as_ref(py)
        .call1((
            date.year(),
            date.month(),
            date.day(),
            hour,
            minute,
            second,
            value.microseconds(),
            types.utc.as_ref(py),
        ))?
        .into())
}
pub fn date_to_python(py: Python<'_>, value: CalendarDate) -> PyResult<PyObject> {
    Ok(types(py)?
        .date
        .as_ref(py)
        .call1((value.year(), value.month(), value.day()))?
        .into())
}
pub fn duration_to_python(py: Python<'_>, value: FixedDuration) -> PyResult<PyObject> {
    Ok(types(py)?
        .duration
        .as_ref(py)
        .call1((value.days(), value.seconds(), value.microseconds()))?
        .into())
}
