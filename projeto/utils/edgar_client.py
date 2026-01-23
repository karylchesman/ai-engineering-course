from typing import Any, Dict

from edgar import Company, set_identity


class EdgarClient:
    # Theses items contain the most relevant information in each form
    # for financial analysis
    FORM_ITEMS = {
        "10-K": ["1", "1A", "7", "8", "9A"],
        "10-Q": ["1", "2", "3", "4"],
    }

    def __init__(self, email: str):
        set_identity(email)

    def fetch_filling_date(self, ticker: str, form_type: str) -> Dict[str, Any]:
        company = Company(ticker)
        filling = company.get_filings(form=form_type).latest()
        metadata = {
            "ticker": ticker,
            "company_name": filling.company,
            "report_date": str(filling.report_date),
            "form_type": filling.form,
        }
        filling_obj = filling.obj()
        items = {}

        for item_num in self.FORM_ITEMS[form_type]:
            item_key = f"Item {item_num}"
            try:
                items[item_key] = filling_obj[item_key]
            except (KeyError, IndexError):
                continue

        return {"metadata": metadata, "items": items}

    def get_combined_text(self, data: Dict) -> str:
        texts = []
        for item_name, item_content in data["items"].items():
            texts.append(f"## {item_name}\n\n{item_content}")

        return "\n\n".join(texts)
