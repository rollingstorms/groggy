//! Checked temporal properties with a lossless Python-compatible microsecond domain.
use crate::{GraphError, GraphResult};
use chrono::{Datelike, Timelike};
use serde::{Deserialize, Serialize};

fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn version(version: u8) -> GraphResult<()> {
    if version == 1 {
        Ok(())
    } else {
        Err(invalid("Unsupported temporal property version"))
    }
}

/// UTC instant, as floor Unix seconds plus a nonnegative microsecond fraction.
/// Leap seconds and years outside 1..=9999 are deliberately excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "InstantParts", into = "InstantParts")]
pub struct UtcInstant {
    seconds: i64,
    microseconds: u32,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InstantParts {
    version: u8,
    seconds: i64,
    microseconds: u32,
}
impl UtcInstant {
    /// Construct a canonical instant without floating-point epoch arithmetic.
    pub fn new(seconds: i64, microseconds: u32) -> GraphResult<Self> {
        chrono::DateTime::from_timestamp(seconds, 0)
            .filter(|date| (1..=9999).contains(&date.year()))
            .ok_or_else(|| invalid("UTC instant must have a year in 1..=9999"))?;
        if microseconds >= 1_000_000 {
            return Err(invalid("Instant microseconds must be below 1000000"));
        }
        Ok(Self {
            seconds,
            microseconds,
        })
    }
    /// Build a UTC instant from checked calendar and clock components.
    pub fn from_calendar(
        date: CalendarDate,
        hour: u32,
        minute: u32,
        second: u32,
        microsecond: u32,
    ) -> GraphResult<Self> {
        let date = chrono::NaiveDate::from_ymd_opt(date.year, date.month, date.day)
            .and_then(|date| date.and_hms_micro_opt(hour, minute, second, microsecond))
            .ok_or_else(|| invalid("Invalid UTC calendar/clock components"))?;
        Self::new(date.and_utc().timestamp(), microsecond)
    }
    /// Floor Unix seconds, including for instants before the epoch.
    pub fn seconds(self) -> i64 {
        self.seconds
    }
    /// Nonnegative fractional microseconds within the second.
    pub fn microseconds(self) -> u32 {
        self.microseconds
    }
    /// Checked UTC calendar/clock components for language adapters.
    pub fn calendar(self) -> GraphResult<(CalendarDate, u32, u32, u32)> {
        let value = chrono::DateTime::from_timestamp(self.seconds, 0)
            .ok_or_else(|| invalid("Invalid UTC instant"))?;
        Ok((
            CalendarDate::new(value.year(), value.month(), value.day())?,
            value.hour(),
            value.minute(),
            value.second(),
        ))
    }
}
impl TryFrom<InstantParts> for UtcInstant {
    type Error = GraphError;
    fn try_from(value: InstantParts) -> GraphResult<Self> {
        version(value.version)?;
        Self::new(value.seconds, value.microseconds)
    }
}
impl From<UtcInstant> for InstantParts {
    fn from(value: UtcInstant) -> Self {
        Self {
            version: 1,
            seconds: value.seconds,
            microseconds: value.microseconds,
        }
    }
}
impl std::fmt::Display for UtcInstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (date, hour, minute, second) = self.calendar().map_err(|_| std::fmt::Error)?;
        write!(
            f,
            "{date}T{hour:02}:{minute:02}:{second:02}.{:06}Z",
            self.microseconds
        )
    }
}

/// A proleptic Gregorian calendar date, independent of any timezone or clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "DateParts", into = "DateParts")]
pub struct CalendarDate {
    year: i32,
    month: u32,
    day: u32,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DateParts {
    version: u8,
    year: i32,
    month: u32,
    day: u32,
}
impl CalendarDate {
    /// Validate a calendar date in the shared Rust/Python year range 1..=9999.
    pub fn new(year: i32, month: u32, day: u32) -> GraphResult<Self> {
        if !(1..=9999).contains(&year)
            || chrono::NaiveDate::from_ymd_opt(year, month, day).is_none()
        {
            return Err(invalid("Invalid calendar date (year must be in 1..=9999)"));
        }
        Ok(Self { year, month, day })
    }
    /// Gregorian year.
    pub fn year(self) -> i32 {
        self.year
    }
    /// Calendar month, 1..=12.
    pub fn month(self) -> u32 {
        self.month
    }
    /// Calendar day, validated against month and leap year.
    pub fn day(self) -> u32 {
        self.day
    }
}
impl TryFrom<DateParts> for CalendarDate {
    type Error = GraphError;
    fn try_from(value: DateParts) -> GraphResult<Self> {
        version(value.version)?;
        Self::new(value.year, value.month, value.day)
    }
}
impl From<CalendarDate> for DateParts {
    fn from(value: CalendarDate) -> Self {
        Self {
            version: 1,
            year: value.year,
            month: value.month,
            day: value.day,
        }
    }
}
impl std::fmt::Display for CalendarDate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// Fixed elapsed duration in canonical timedelta form; a day is exactly 86400s.
/// Calendar months/years and timezone-dependent periods are not durations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "DurationParts", into = "DurationParts")]
pub struct FixedDuration {
    days: i32,
    seconds: u32,
    microseconds: u32,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DurationParts {
    version: u8,
    days: i32,
    seconds: u32,
    microseconds: u32,
}
impl FixedDuration {
    /// Validate canonical signed days with nonnegative second/microsecond remainders.
    pub fn new(days: i32, seconds: u32, microseconds: u32) -> GraphResult<Self> {
        if !(-999_999_999..=999_999_999).contains(&days)
            || seconds >= 86_400
            || microseconds >= 1_000_000
        {
            return Err(invalid("Duration requires days in +/-999999999, seconds below 86400 and microseconds below 1000000"));
        }
        Ok(Self {
            days,
            seconds,
            microseconds,
        })
    }
    /// Signed normalized days.
    pub fn days(self) -> i32 {
        self.days
    }
    /// Nonnegative seconds within the day.
    pub fn seconds(self) -> u32 {
        self.seconds
    }
    /// Nonnegative microseconds within the second.
    pub fn microseconds(self) -> u32 {
        self.microseconds
    }
}
impl TryFrom<DurationParts> for FixedDuration {
    type Error = GraphError;
    fn try_from(value: DurationParts) -> GraphResult<Self> {
        version(value.version)?;
        Self::new(value.days, value.seconds, value.microseconds)
    }
}
impl From<FixedDuration> for DurationParts {
    fn from(value: FixedDuration) -> Self {
        Self {
            version: 1,
            days: value.days,
            seconds: value.seconds,
            microseconds: value.microseconds,
        }
    }
}
impl std::fmt::Display for FixedDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}d+{}.{:06}s",
            self.days, self.seconds, self.microseconds
        )
    }
}
