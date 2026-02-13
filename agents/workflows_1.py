import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"]
)

model_name = "meta-llama/llama-4-scout-17b-16e-instruct"


class ExtractEvent(BaseModel):
    description: str = Field(description="Descrição do evento extraído do texto.")
    is_calendar_event: bool = Field(
        description="Indica se o evento é um evento de calendário."
    )
    confidence_score: float = Field(
        description="Pontuação de confiança na extração do evento entre 0 e 1."
    )


class EventDetails(BaseModel):
    name: str = Field(description="Nome do evento.")
    date: str = Field(
        description="Data e hora do evento no formato ISO 8601, por exemplo, '2024-07-01T14:00:00Z'."
    )
    duration_minutes: Optional[int] = Field(
        None, description="Duração esperada em minutos."
    )
    participants: list[str] = Field(description="Lista de participantes.")


class EventConfirmation(BaseModel):
    message_confirmation: str = Field(
        description="Mensagem de confirmação em linguagem natural."
    )
    link_to_calendar: Optional[str] = Field(
        None, description="Link para o evento no calendário, se aplicável."
    )


def extract_event_info(user_input: str) -> ExtractEvent:
    today = datetime.now().strftime("%A, %d de %B de %Y")
    date_context = f"Hoje é {today}."
    response = client.responses.parse(
        model=model_name,
        input=f"{date_context} analise se o texto descreve um evento de calendário.",
        instructions=f"Extraia as informações do evento do seguinte texto: '{user_input}'",
        text_format=ExtractEvent,
    )
    return response.output_parsed


def analyze_event_details(description: str) -> EventDetails:
    today = datetime.now().strftime("%A, %d de %B de %Y")
    date_context = f"Hoje é {today}."
    response = client.responses.parse(
        model=model_name,
        input=f"""{date_context} Extraia informações detalhadas do evento. Quando as 
        datas fizerem referência a 'amanhã', 'próxima semana', etc., use a data atual 
        como referência.""",
        instructions=f"Extraia detalhes estruturados deste texto de evento: '{description}'",
        text_format=EventDetails,
    )
    return response.output_parsed


def generate_confirmation(details: EventDetails) -> EventConfirmation:
    response = client.responses.parse(
        model=model_name,
        input="""Você é um assistente de calendário.
        Sua tarefa é gerar uma mensagem solicitando que o usuário confirme os dados de um evento antes que ele seja criado.

        A mensagem deve:

        Informar que o evento ainda não foi criado;

        Listar claramente os dados fornecidos (título, data, horário, local, descrição, etc.);

        Pedir confirmação explícita do usuário para prosseguir com a criação;

        Perguntar se deseja alterar algum dado antes da confirmação.

        Use um tom claro, objetivo e amigável.""",
        instructions=f"""Crie uma confirmação para este evento: {details.model_dump()}""",
        text_format=EventConfirmation,
    )
    return response.output_parsed


def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    initial_extraction = extract_event_info(user_input)
    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confidence_score < 0.7
    ):
        return None

    event_details = analyze_event_details(initial_extraction.description)
    confirmation = generate_confirmation(event_details)
    return confirmation


user_input = """Vamos fazer uma transmissão ao vivo na 
próxima segunda-feira às 20h com Daniel e Alberto para apresentar 
o lançamento do novo curso, deve durar umas 2 horas.
"""

confirmation = process_calendar_request(user_input)
if confirmation:
    print("Evento confirmado:")
    print(confirmation.message_confirmation)
    if confirmation.link_to_calendar:
        print(f"Link para o calendário: {confirmation.link_to_calendar}")
else:
    print("Nenhum evento de calendário identificado ou confiança insuficiente.")


# user_input = """Você pode enviar um e-mail para Daniel e Alberto
# para discutir o roteiro do projeto?
# """
# confirmation = process_calendar_request(user_input)
# if confirmation:
#     print(f"Confirmação: {confirmation.message_confirmation}")
#     if confirmation.link_to_calendar:
#         print(f"Link do calendário: {confirmation.link_to_calendar}")
# else:
#     print("Isto não parece ser uma solicitação de evento de calendário")
