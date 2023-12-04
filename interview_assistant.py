import argparse
from typing import Optional
import json
from dotenv import dotenv_values
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema.runnable import ConfigurableField
from langchain.chat_models import ChatOpenAI
from instructions import (
    generic_instruct_ctx,
    bug_instruct_ctx,
    mode_instruct_ctx,
    concept_instruct_ctx,
    direct_q,
    intro,
)


class InterviewAssistant:
    def __init__(
        self,
        coding_question: str,
        code_solution: str,
        api_key: str,
        initial_mode: str = "conceptual",
        mode_switching: str = "heuristic",
        heuristic_switchover=None,
        max_token_args=None,
    ):
        self.api_key = api_key
        self.mode = initial_mode
        self.coding_q = coding_question
        self.solution = code_solution
        self.mode_switching = mode_switching
        self.num_invocations = 0
        self.temperature = 0.8
        self.model = "gpt-3.5-turbo-1106"
        self.chat = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=250,
            api_key=self.api_key,
        ).configurable_fields(
            max_tokens=ConfigurableField(
                id="max_tokens",
                name="Max Tokens",
                description="Maximum Tokens to output",
            )
        )
        if heuristic_switchover is None:
            self.heuristic_switchover = 3
        else:
            self.heuristic_switchover = heuristic_switchover
        if (
            max_token_args is None
            or "hint_max" not in max_token_args
            or "direct_question_response_max" not in max_token_args
        ):
            self.max_token_args = {"hint_max": 250, "direct_question_response_max": 500}
        else:
            self.max_token_args = max_token_args
        # Stop words
        self.stop_words = [
            "Question:",
            "Current Code:",
            "Current Transcript:",
            "Solution:",
        ]

    def __call__(
        self,
        current_code: str,
        current_transcript: str,
        question: Optional[str] = None,
        direct_question_flg: str = False,
    ):
        if direct_question_flg:
            if question is None:
                raise ValueError(
                    "No explicit Question has Been Passed in " "But Direct Flag is here"
                )
            return self.direct_question_response(
                current_code, current_transcript, question
            )
        self.num_invocations += 1
        # Elicit either a conceptual, high-level hint or a fine-grained hint
        if self.mode != "generic":
            self.mode = self.determine_mode(current_code, current_transcript)
        if self.mode == "conceptual":
            instruction = concept_instruct_ctx
        elif self.mode == "fine-grained":
            instruction = bug_instruct_ctx
        else:
            instruction = generic_instruct_ctx
        print(f"Instruction Used: {instruction} ")
        return self.generate_chat_response(
            instruction,
            coding_question=self.coding_q,
            code_snippet=current_code,
            interview_transcript=current_transcript,
            solution=self.solution,
            max_tokens=self.max_token_args["hint_max"],
        )

    def generate_chat_response(
        self,
        sys_instruction: str,
        coding_question: str,
        code_snippet: str,
        interview_transcript: str,
        solution: str = "",
        max_tokens: int = 250,
    ):
        # Construct user prompt
        user_prompt = f"Question: {coding_question}\nCurrent Code: {code_snippet}\nCurrent Transcript: {interview_transcript}"
        if solution:
            user_prompt += f"\nSolution: {solution}"
        messages = [
            SystemMessage(content=sys_instruction),
            HumanMessage(content=user_prompt),
        ]
        ai_response = self.chat.with_config(
            configurable={"max_tokens": max_tokens}
        ).invoke(messages, stop=self.stop_words)
        return ai_response.content

    def determine_mode(self, code, transcript):
        if self.mode_switching == "heuristic":
            if self.num_invocations < self.heuristic_switchover:
                return "conceptual"
            else:
                return "fine-grained"
        else:
            llm_response = self.generate_chat_response(
                mode_instruct_ctx,
                coding_question=self.coding_q,
                code_snippet=code,
                interview_transcript=transcript,
                solution=self.solution,
                max_tokens=250,
            )
            if "yes" in llm_response.lower():
                return "conceptual"
            else:
                return "fine-grained"

    def direct_question_response(self, code, transcript, question) -> str:
        # Construct user prompt
        user_prompt = f"Question: {self.coding_q}\nCurrent Code: {code}\nCurrent Transcript: {transcript}"
        user_prompt += f"\nSolution: {self.solution}\nStudent Question: {question}"
        messages = [SystemMessage(content=direct_q), HumanMessage(content=user_prompt)]
        ai_response = self.chat.with_config(
            configurable={
                "max_tokens": self.max_token_args["direct_question_response_max"]
            }
        ).invoke(messages, stop=self.stop_words)
        return ai_response.content

    def check_hint(self, hint, true_ans) -> bool:
        raise NotImplementedError

    def upgrade_model(self):
        self.model = "gpt-4-1106-preview"
        self.chat = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=250,
            api_key=self.api_key,
        ).configurable_fields(
            max_tokens=ConfigurableField(
                id="max_tokens",
                name="Max Tokens",
                description="Maximum Tokens to output",
            )
        )
        self.mode = "generic"

    def override_mode(self, mode):
        self.mode = mode

    def get_current_mode(self):
        return self.mode
