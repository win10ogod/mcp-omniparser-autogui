#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import asyncio
from contextlib import redirect_stdout
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage

from langchain_openai import ChatOpenAI

from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

from mcp_autogui.langchain.mcp_manager import McpManager
from mcp_autogui.langchain.agent_graph import create_agent_graph

load_dotenv()

def create_agent(tools):
    #llm = ChatOpenAI(model="gpt-4o-mini")
    harm_block = HarmBlockThreshold.BLOCK_NONE
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: harm_block,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: harm_block,
        HarmCategory.HARM_CATEGORY_HARASSMENT: harm_block,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: harm_block,
    }
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash-exp',
        safety_settings=safety_settings,
    )
    agent = create_agent_graph(llm, tools, debug=True)
    return agent

async def batch_main(prompts: list[str], system_prompt: str = ''):
    with redirect_stdout(sys.stderr):
        ret = []

        mcp_manager = McpManager()
        await mcp_manager.load('langchain_settings/mcp_config.json')
        agent = create_agent(mcp_manager.get_tools())

        for prompt in prompts:
            chat_history = ChatMessageHistory()
            if system_prompt == '':
                system_prompt = '''You are using a Windows device.
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.

You may be given some history plan and actions, this is the response from the previous loop.
You should carefully consider your plan base on the task, screenshot, and history actions.

IMPORTANT NOTES:
1. You should only give a single action at a time.
2. You should give an analysis to the current screen, and reflect on what has been done by looking at the history, then describe your step-by-step thoughts on how to achieve the task.
3. The tasks involve buying multiple products or navigating through multiple pages. You should break it into subgoals and complete each subgoal one by one in the order of the instructions.
4. avoid choosing the same action/elements multiple times in a row, if it happens, reflect to yourself, what may have gone wrong, and predict a different action.
5. If you are prompted with login information page or captcha page, or you think it need user's permission to do the next action, finish actions and ask the user for a response.
6. Exit if you have achieved your goal.'''
            chat_history.add_message(SystemMessage(content=system_prompt))
            chat_history.add_user_message(prompt)

            print(prompt)

            response = await agent.ainvoke(
                {"messages": chat_history.messages},
                {"recursion_limit": 100},
            )

            ret.append(response["messages"][-1].content)
            print(response["messages"][-1].content)

        mcp_manager.stop_servers()

        return ret

if __name__ == "__main__":
    prompts = [
        '''画面を確認し、ブラウザから「MCPサーバー」と入力して検索してください''',
    ]
    asyncio.run(batch_main(prompts))
