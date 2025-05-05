import chainlit as cl
import random


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Which are HumAIne's AI paradigms?",
            message="Which are HumAIne's AI paradigms?",
            icon="/public/intelligent-assistant.png",
        ),
        cl.Starter(
            label="What kind of data have I available?",
            message="What kind of data have I available?",
            icon="/public/s_data.png",
            ),
        cl.Starter(
            label="Are there any ready-to-use Pipelines for my data?",
            message="Are there any ready-to-use Pipelines for my data?",
            icon="/public/s_pipeline.png",
            ),
       
    ]
