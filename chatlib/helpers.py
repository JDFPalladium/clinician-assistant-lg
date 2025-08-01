import dateparser
import dateparser.search
from datetime import datetime
from dateutil.relativedelta import relativedelta

RELATIVE_INDICATORS = [
    "ago",
    "later",
    "before",
    "after",
    "yesterday",
    "tomorrow",
    "today",
    "tonight",
    "last",
    "next",
    "this",
    "coming",
    "previous",
    "past",
]


def is_relative_date(text_relative):
    text_lower = text_relative.lower()
    return any(word in text_lower for word in RELATIVE_INDICATORS)


def dateparser_detect(text_dates):
    results_date = dateparser.search.search_dates(text_dates, languages=["en"])
    if not results_date:
        return []
    filtered = [r for r in results_date if not is_relative_date(r[0])]
    return filtered


def describe_relative_date(dt, reference=None):
    if reference is None:
        reference = datetime.now()

    delta = relativedelta(reference, dt)

    if delta.years > 0:
        return f"{delta.years} year{'s' if delta.years > 1 else ''} ago"
    elif delta.months > 0:
        return f"{delta.months} month{'s' if delta.months > 1 else ''} ago"
    elif delta.days >= 7:
        weeks = delta.days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    elif delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    else:
        return "today"
