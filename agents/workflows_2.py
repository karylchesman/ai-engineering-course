import os
from datetime import datetime
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"]
)

model_name = "meta-llama/llama-4-scout-17b-16e-instruct"


class CalendarRequestType(BaseModel):
    request_type: Literal["new_event", "update_event", "other"] = Field(
        description="Tipo de solicitação de calendário sendo feita."
    )
    confidence_score: float = Field(description="Pontuação de confiança entre 0 e 1.")
    description: str = Field(description="Descrição limpa da solicitação.")


class DetailsOfNewEvent(BaseModel):
    name: str = Field(description="Nome do evento.")
    date: str = Field(description="Data e hora do evento no formato ISO 8601'.")
    duration_minutes: Optional[int] = Field(
        None, description="Duração esperada em minutos."
    )
    participants: list[str] = Field(description="Lista de participantes.")


class RequestedChange(BaseModel):
    field_changed: str = Field(description="Campo a ser alterado.")
    new_value: str = Field(description="Novo valor para o campo alterado.")


class EventModificationDetails(BaseModel):
    event_identifier: str = Field(
        description="Descrição para identificar o evento existente."
    )
    changes_requested: list[RequestedChange] = Field(
        description="Lista de mudanças solicitadas para o evento."
    )
    participantes_to_add: list[str] = Field(
        description="Lista de participantes a serem adicionados, se aplicável."
    )
    participantes_to_remove: list[str] = Field(
        description="Lista de participantes a serem removidos, se aplicável."
    )


class CalendarAnswer(BaseModel):
    success: bool = Field(description="Se a operação foi bem-sucedida.")
    message: str = Field(description="Mensagem de resposta amigável ao usuário.")
    link_to_calendar: Optional[str] = Field(
        None, description="Link para o evento no calendário, se aplicável."
    )


def classify_calendar_request(user_input: str) -> CalendarRequestType:
    today = datetime.now().strftime("%A, %d de %B de %Y")
    date_context = f"Hoje é {today}."
    response = client.responses.parse(
        model=model_name,
        input=f"{date_context} classifique o tipo de solicitação de calendário.",
        instructions=f"Classifique a seguinte solicitação de calendário: '{user_input}'",
        text_format=CalendarRequestType,
    )
    return response.output_parsed


def process_new_event_details(description: str) -> CalendarAnswer:
    today = datetime.now().strftime("%A, %d de %B de %Y")
    date_context = f"Hoje é {today}."
    response = client.responses.parse(
        model=model_name,
        input=f"{date_context} Extraia detalhes estruturados para criar um novo evento.",
        instructions=f"Extraia detalhes estruturados deste texto de evento: '{description}'",
        text_format=DetailsOfNewEvent,
    )
    if response.output_parsed is None:
        return CalendarAnswer(
            success=False,
            message="Desculpe, não consegui extrair detalhes suficientes para criar o evento. Por favor, forneça mais informações.",
        )
    details = response.output_parsed
    return CalendarAnswer(
        success=True,
        message=f"Evento '{details.name}' criado para {details.date} com duração de {details.duration_minutes} minutos e participantes: {', '.join(details.participants)}.",
        link_to_calendar=f"calendar://new?event={details.name}",
    )


def process_event_modification(description: str) -> CalendarAnswer:
    today = datetime.now().strftime("%A, %d de %B de %Y")
    date_context = f"Hoje é {today}."
    response = client.responses.parse(
        model=model_name,
        input=f"{date_context} Extraia detalhes estruturados para modificar um evento existente.",
        instructions=f"Extraia detalhes estruturados deste texto de modificação de evento: '{description}'",
        text_format=EventModificationDetails,
    )
    if response.output_parsed is None:
        return CalendarAnswer(
            success=False,
            message="Desculpe, não consegui extrair detalhes suficientes para modificar o evento. Por favor, forneça mais informações.",
        )
    modification_details = response.output_parsed
    changes_summary = "; ".join(
        [
            f"{change.field_changed} para '{change.new_value}'"
            for change in modification_details.changes_requested
        ]
    )

    return CalendarAnswer(
        success=True,
        message=f"Modificado evento '{modification_details.event_identifier}': {changes_summary}",
        link_to_calendar=f"calendar://modify?event={modification_details.event_identifier}",
    )


def process_calendar_request(user_input: str) -> CalendarAnswer:
    classification = classify_calendar_request(user_input)

    if classification.confidence_score < 0.7:
        return CalendarAnswer(
            success=False,
            message="Desculpe, não consegui classificar claramente o tipo de solicitação de calendário. Por favor, reformule sua solicitação.",
        )

    if classification.request_type == "create":
        return process_new_event_details(classification.description)
    elif classification.request_type == "update":
        return process_event_modification(classification.description)
    else:
        return CalendarAnswer(
            success=False,
            message="Desculpe, não consegui classificar o tipo de solicitação de calendário. Por favor, reformule sua solicitação.",
        )


user_input = "Vamos agendar uma reunião de equipe na próxima terça-feira às 14h com Daniel e Alberto"
resultado = process_calendar_request(user_input)
if resultado:
    print(f"Resposta: {resultado.message}")
else:
    print("Solicitação não reconhecida como operação de calendário")


modify_user_input = (
    "Você pode mover a reunião de equipe com Daniel e Alberto para quarta-feira às 15h?"
)
resultado = process_calendar_request(modify_user_input)
if resultado:
    print(f"Resposta: {resultado.message}")
else:
    print("Solicitação não reconhecida como operação de calendário")


invalid_user_input = "Como está o clima hoje?"
resultado = process_calendar_request(invalid_user_input)
print(f"Resposta: {resultado.message}")
