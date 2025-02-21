from typing import cast
from langgraph.func import entrypoint,task
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv,find_dotenv
from pydantic import BaseModel,Field
import os

load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",google_api_key=os.getenv("GEMINI_API_KEY"))


# Evaluator Optimizer workflow
# POem flow == Funny Poem Flow
# Input == Poem Topic
# GENERATOR: Generate a Funny poem about the topic
# EVALUATOR: Evaluate the poem if it is funny then else retry
# OPTIMIZER: Optimize the poem
# Output == Optimized Poem





@task
def poem_generator(input_data:str)->str:
    """Generate a poem about the topic"""
    return llm.invoke(f"Generate a Funny poem about : {input_data}").content

@task
def poem_evaluator(input_data:str)->str:
    # Output Must be Funny Or Retry || Structured Output
    """Call the evaluator to get the report"""
    return llm.invoke(f"Evaluate the poem if its funny or not : {input_data} Return a number between 0 and 10 based on how funny the poem is if its funny return 0 only return number").content



@entrypoint()
def evaluator_optimizer_workflow(topic:str)->str:
    #step 1: call the generator
    while True:
        poem = poem_generator(topic).result()
        print(f"Generated Poem: {poem}")
        evaluator_report = poem_evaluator(poem).result()
        print(f"Evaluator Report: {evaluator_report}")
        if int(evaluator_report) > 5:
            print("Poem is Funny")
            break
        else:
            print("Poem is not Funny")
            continue
    return poem



def main():
    poem = evaluator_optimizer_workflow.invoke("Vertical AI Agents")
    print(f"\n\nOptimized Poem: {poem}")
    with open("poem.md","w") as f:
        f.write(poem)
    print("Poem saved to poem.md")




