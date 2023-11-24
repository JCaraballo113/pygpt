import os
from dotenv import load_dotenv
from langchain.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--language",
    type=str,
    default="python",
    help="The language of the code",
)

parser.add_argument(
    "--task",
    type=str,
    default="print hello world",
    help="The task of the code",
)

args = parser.parse_args()

# Load environment variables from .env file
load_dotenv()
llm = openai.OpenAI()

code_template = PromptTemplate(
    input_variables=["task", "language"],
    template="Write a very short {language} code that will {task}",
    args=dict(language=args.language, task=args.task),
)

test_template = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n {code}",
)

code_chain = LLMChain(llm=llm, prompt=code_template, output_key="code")
code_chain({"task": args.task, "language": args.language})

test_chain = LLMChain(llm=llm, prompt=test_template, output_key="test")

chain = SequentialChain(chains=[code_chain, test_chain], input_variables=[
                        "task", "language"], output_variables=["code", "test"])

result = chain({"task": args.task, "language": args.language})
print(result["test"])
