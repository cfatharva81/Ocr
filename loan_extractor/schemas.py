"""Pydantic v2 models for the extracted loan application.

Every model normalizes raw extracted strings in field_validator(mode="before"):
"$4,750.01" -> 4750.01, dates -> ISO date, phones -> digits, "Yes"/checked -> True.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import normalizers as N

T = TypeVar("T")

Citizenship = Literal["U.S. Citizen", "Permanent Resident Alien", "Non-Permanent Resident Alien"]
MaritalStatus = Literal["Married", "Separated", "Unmarried"]
HousingType = Literal["No primary housing expense", "Own", "Rent"]
LoanPurpose = Literal["Purchase", "Refinance", "Other"]
OccupancyType = Literal["Primary Residence", "Second Home", "Investment Property",
                        "FHA Secondary Residence"]
CreditType = Literal["Individual", "Joint"]


def _choice(value, allowed: tuple[str, ...]):
    """Map free text onto one of the allowed literals (case/spacing/OCR tolerant)."""
    if value is None:
        return None
    from rapidfuzz import fuzz, process
    t = N.clean_text(str(value))
    if not t:
        return None
    for a in allowed:
        if t.casefold() == a.casefold():
            return a
    hit = process.extractOne(t, allowed, scorer=fuzz.ratio, score_cutoff=80)
    return hit[0] if hit else t      # let Literal validation reject it


class Extracted(BaseModel, Generic[T]):
    """One extracted value with its provenance."""
    value: Optional[T] = None
    confidence: float = 0.0
    page: Optional[int] = None
    source_text: Optional[str] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    strategy_used: Optional[str] = None


class _Base(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str) and not N.clean_text(v):
            return None
        return v


def _text(v):
    return N.clean_text(v) if isinstance(v, str) else v


def _blank(v) -> bool:
    """None or an empty/whitespace string. Subclass validators run before the base
    blank->None validator, so every raising validator checks this first."""
    return v is None or (isinstance(v, str) and N.clean_text(v) is None)


# --------------------------------------------------------------- section 1

class BorrowerInformation(_Base):
    borrower_name: str
    social_security_number: str
    date_of_birth: date
    citizenship_status: Optional[Citizenship] = None
    alternate_names: Optional[str] = None
    credit_type: Optional[CreditType] = None
    total_borrowers: Optional[int] = Field(default=None, ge=1, le=20)
    marital_status: Optional[MaritalStatus] = None
    number_of_dependents: Optional[int] = Field(default=None, ge=0, le=30)

    @field_validator("borrower_name", "alternate_names", mode="before")
    @classmethod
    def _names(cls, v):
        return _text(v)

    @field_validator("social_security_number", mode="before")
    @classmethod
    def _ssn(cls, v):
        if _blank(v):
            return None
        s = N.ssn(v)
        if s is None:
            raise ValueError("SSN must contain 9 digits")
        return s

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def _dob(cls, v):
        if _blank(v):
            return None
        d = N.to_date(v)
        if d is None:
            raise ValueError(f"unparseable date: {v!r}")
        return d

    @field_validator("citizenship_status", mode="before")
    @classmethod
    def _cit(cls, v):
        return _choice(v, Citizenship.__args__)

    @field_validator("marital_status", mode="before")
    @classmethod
    def _mar(cls, v):
        return _choice(v, MaritalStatus.__args__)

    @field_validator("credit_type", mode="before")
    @classmethod
    def _credit(cls, v):
        return _choice(v, CreditType.__args__)

    @field_validator("total_borrowers", "number_of_dependents", mode="before")
    @classmethod
    def _ints(cls, v):
        return N.integer(v)


class ContactInformation(_Base):
    home_phone: Optional[str] = None
    cell_phone: Optional[str] = None
    work_phone: Optional[str] = None
    email: Optional[str] = None

    @field_validator("home_phone", "cell_phone", "work_phone", mode="before")
    @classmethod
    def _phone(cls, v):
        if _blank(v):
            return None
        p = N.phone(v)
        if p is None:
            raise ValueError(f"not a phone number: {v!r}")
        return p

    @field_validator("email", mode="before")
    @classmethod
    def _email(cls, v):
        e = N.email(v)
        if e is not None and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", e):
            raise ValueError(f"not an email address: {v!r}")
        return e


class AddressInformation(_Base):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    country: Optional[str] = None
    years_at_address: Optional[int] = Field(default=None, ge=0, le=99)
    months_at_address: Optional[int] = Field(default=None, ge=0, le=11)
    housing_type: Optional[HousingType] = None

    @field_validator("street", "city", "country", mode="before")
    @classmethod
    def _txt(cls, v):
        return _text(v)

    @field_validator("state", mode="before")
    @classmethod
    def _state(cls, v):
        if _blank(v):
            return None
        s = N.state_code(v)
        if s is None:
            raise ValueError(f"unknown state: {v!r}")
        return s

    @field_validator("zip", mode="before")
    @classmethod
    def _zip(cls, v):
        if _blank(v):
            return None
        z = N.zip_code(v)
        if z is None:
            raise ValueError(f"invalid ZIP: {v!r}")
        return z

    @field_validator("years_at_address", "months_at_address", mode="before")
    @classmethod
    def _ints(cls, v):
        return N.integer(v)

    @field_validator("housing_type", mode="before")
    @classmethod
    def _housing(cls, v):
        return _choice(v, HousingType.__args__)


class EmploymentInformation(_Base):
    employer_name: Optional[str] = None
    employer_street: Optional[str] = None
    employer_city: Optional[str] = None
    employer_state: Optional[str] = None
    employer_zip: Optional[str] = None
    position_title: Optional[str] = None
    employment_start_date: Optional[date] = None
    years_in_profession: Optional[int] = Field(default=None, ge=0, le=80)
    months_in_profession: Optional[int] = Field(default=None, ge=0, le=11)

    @field_validator("employer_name", "employer_street", "employer_city", "position_title",
                     mode="before")
    @classmethod
    def _txt(cls, v):
        return _text(v)

    @field_validator("employer_state", mode="before")
    @classmethod
    def _state(cls, v):
        return AddressInformation._state(v)

    @field_validator("employer_zip", mode="before")
    @classmethod
    def _zip(cls, v):
        return AddressInformation._zip(v)

    @field_validator("employment_start_date", mode="before")
    @classmethod
    def _date(cls, v):
        return BorrowerInformation._dob(v)

    @field_validator("years_in_profession", "months_in_profession", mode="before")
    @classmethod
    def _ints(cls, v):
        return N.integer(v)


def _money(v):
    if _blank(v):
        return None
    m = N.currency(v)
    if m is None:
        raise ValueError(f"not an amount: {v!r}")
    return m


class IncomeInformation(_Base):
    base_income: Optional[float] = None
    overtime_income: Optional[float] = None
    bonus_income: Optional[float] = None
    commission_income: Optional[float] = None
    military_income: Optional[float] = None
    other_income: Optional[float] = None
    total_monthly_income: Optional[float] = None

    @field_validator("*", mode="before")
    @classmethod
    def _amounts(cls, v):
        return _money(v)


# --------------------------------------------------------------- section 2

class Asset(_Base):
    account_type: Optional[str] = None
    financial_institution: Optional[str] = None
    account_number: Optional[str] = None
    market_value: Optional[float] = None

    @field_validator("account_type", "financial_institution", mode="before")
    @classmethod
    def _txt(cls, v):
        return _text(v)

    @field_validator("account_number", mode="before")
    @classmethod
    def _acct(cls, v):
        return N.account_number(v)

    @field_validator("market_value", mode="before")
    @classmethod
    def _val(cls, v):
        return _money(v)


class AssetsInformation(_Base):
    accounts: list[Asset] = Field(default_factory=list)
    total_assets: Optional[float] = None

    @field_validator("total_assets", mode="before")
    @classmethod
    def _total(cls, v):
        return _money(v)


class OtherAsset(_Base):
    asset_type: Optional[str] = None
    market_value: Optional[float] = None

    @field_validator("asset_type", mode="before")
    @classmethod
    def _txt(cls, v):
        return _text(v)

    @field_validator("market_value", mode="before")
    @classmethod
    def _val(cls, v):
        return _money(v)


class OtherAssets(_Base):
    other_assets_present: Optional[bool] = None
    other_assets: list[OtherAsset] = Field(default_factory=list)

    @field_validator("other_assets_present", mode="before")
    @classmethod
    def _bool(cls, v):
        return N.boolean(v)


class Liability(_Base):
    account_type: Optional[str] = None
    company_name: Optional[str] = None
    account_number: Optional[str] = None
    unpaid_balance: Optional[float] = None
    monthly_payment: Optional[float] = None
    paid_before_closing: Optional[bool] = None

    @field_validator("account_type", "company_name", mode="before")
    @classmethod
    def _txt(cls, v):
        return _text(v)

    @field_validator("account_number", mode="before")
    @classmethod
    def _acct(cls, v):
        return N.account_number(v)

    @field_validator("unpaid_balance", "monthly_payment", mode="before")
    @classmethod
    def _val(cls, v):
        return _money(v)

    @field_validator("paid_before_closing", mode="before")
    @classmethod
    def _bool(cls, v):
        return N.boolean(v)


class LiabilitySummary(_Base):
    total_unpaid_balance: Optional[float] = None
    total_monthly_debt_payment: Optional[float] = None

    @field_validator("*", mode="before")
    @classmethod
    def _val(cls, v):
        return _money(v)


# --------------------------------------------------------------- section 3/4

class RealEstateInformation(_Base):
    owns_real_estate: Optional[bool] = None
    property_count: Optional[int] = Field(default=None, ge=0)

    @field_validator("owns_real_estate", mode="before")
    @classmethod
    def _bool(cls, v):
        return N.boolean(v)

    @field_validator("property_count", mode="before")
    @classmethod
    def _int(cls, v):
        return N.integer(v)


class LoanInformation(_Base):
    loan_amount: float = Field(gt=0)
    loan_purpose: Optional[LoanPurpose] = None
    property_street: Optional[str] = None
    property_city: Optional[str] = None
    property_state: Optional[str] = None
    property_zip: Optional[str] = None
    property_county: Optional[str] = None
    occupancy_type: Optional[OccupancyType] = None
    number_of_units: Optional[int] = Field(default=None, ge=1, le=4)
    property_value: float = Field(gt=0)

    @field_validator("loan_amount", "property_value", mode="before")
    @classmethod
    def _money(cls, v):
        return _money(v)

    @field_validator("loan_purpose", mode="before")
    @classmethod
    def _purpose(cls, v):
        return _choice(v, LoanPurpose.__args__)

    @field_validator("occupancy_type", mode="before")
    @classmethod
    def _occ(cls, v):
        return _choice(v, OccupancyType.__args__)

    @field_validator("property_street", "property_city", "property_county", mode="before")
    @classmethod
    def _txt(cls, v):
        return _text(v)

    @field_validator("property_state", mode="before")
    @classmethod
    def _state(cls, v):
        return AddressInformation._state(v)

    @field_validator("property_zip", mode="before")
    @classmethod
    def _zip(cls, v):
        return AddressInformation._zip(v)

    @field_validator("number_of_units", mode="before")
    @classmethod
    def _int(cls, v):
        return N.integer(v)


class PropertyDetails(_Base):
    mixed_use_property: Optional[bool] = None
    manufactured_home: Optional[bool] = None

    @field_validator("*", mode="before")
    @classmethod
    def _bool(cls, v):
        return N.boolean(v)


class GiftInformation(_Base):
    gift_type: Optional[str] = None
    deposited: Optional[bool] = None
    source: Optional[str] = None
    cash_or_market_value: Optional[float] = None

    @field_validator("gift_type", "source", mode="before")
    @classmethod
    def _txt(cls, v):
        return _text(v)

    @field_validator("deposited", mode="before")
    @classmethod
    def _bool(cls, v):
        return N.boolean(v)

    @field_validator("cash_or_market_value", mode="before")
    @classmethod
    def _val(cls, v):
        return _money(v)


# --------------------------------------------------------------- top level

class LoanApplication(_Base):
    borrower_information: BorrowerInformation
    contact_information: ContactInformation
    address_information: AddressInformation
    employment_information: EmploymentInformation
    income_information: IncomeInformation
    assets: AssetsInformation
    other_assets: OtherAssets
    liabilities: list[Liability] = Field(default_factory=list)
    liability_summary: LiabilitySummary
    real_estate_information: RealEstateInformation
    loan_information: LoanInformation
    property_details: PropertyDetails
    gift_information: list[GiftInformation] = Field(default_factory=list)


# Output group name -> model used to validate it (list groups use the item model).
GROUP_MODELS: dict[str, type[BaseModel]] = {
    "borrower_information": BorrowerInformation,
    "contact_information": ContactInformation,
    "address_information": AddressInformation,
    "employment_information": EmploymentInformation,
    "income_information": IncomeInformation,
    "real_estate_information": RealEstateInformation,
    "loan_information": LoanInformation,
    "property_details": PropertyDetails,
    "liability_summary": LiabilitySummary,
}
ROW_MODELS: dict[str, type[BaseModel]] = {
    "assets": Asset,
    "other_assets": OtherAsset,
    "liabilities": Liability,
    "gift_information": GiftInformation,
}
OUTPUT_KEYS = ["borrower_information", "contact_information", "address_information",
               "employment_information", "income_information", "assets", "other_assets",
               "liabilities", "liability_summary", "real_estate_information", "loan_information",
               "property_details", "gift_information"]


def required_fields(model: type[BaseModel]) -> list[str]:
    return [n for n, f in model.model_fields.items() if f.is_required()]


def dump_value(v: Any) -> Any:
    if isinstance(v, date):
        return v.isoformat()
    return v
