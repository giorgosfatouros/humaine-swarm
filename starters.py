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
            label="Check out our ML Pipelines",
            message="Check out our ready-to-use ML Pipelines",
            icon="/public/s_pipeline.png",
            ),
       
    ]
