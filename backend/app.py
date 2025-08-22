from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import os
import json
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

app = Flask(__name__)
CORS(app)


class AgentCreator:
    def __init__(self):
        self.search_tool = SerperDevTool()

    def parse_prompt(self, prompt):
        """Parse user prompt to determine agent type and requirements"""
        prompt_lower = prompt.lower()

        # Determine agent category
        if any(word in prompt_lower for word in ['temperature', 'sensor', 'gpio', 'led', 'raspberry', 'hardware']):
            category = 'hardware'
        elif any(word in prompt_lower for word in ['web', 'scrape', 'website', 'data', 'api']):
            category = 'web'
        elif any(word in prompt_lower for word in ['email', 'notification', 'alert', 'message']):
            category = 'communication'
        else:
            category = 'general'

        return category

    def create_agent_config(self, prompt, category):
        """Create CrewAI agent configuration based on prompt"""

        configs = {
            'hardware': {
                'role': 'Hardware Control Specialist',
                'goal': f'Execute hardware-related tasks: {prompt}',
                'backstory': 'You are an expert in embedded systems and IoT devices. You excel at controlling sensors, actuators, and GPIO interfaces on Raspberry Pi and similar hardware platforms.',
                'tools': [],
                'code_template': '''
import RPi.GPIO as GPIO
import time
from datetime import datetime

class HardwareAgent:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)

    def execute_task(self):
        """Generated based on user prompt"""
        # Hardware control logic here
        print(f"Hardware agent executing: {prompt}")
        return "Hardware task completed"

    def cleanup(self):
        GPIO.cleanup()

if __name__ == "__main__":
    agent = HardwareAgent()
    try:
        result = agent.execute_task()
        print(result)
    finally:
        agent.cleanup()
'''
            },
            'web': {
                'role': 'Web Data Specialist',
                'goal': f'Handle web-related tasks: {prompt}',
                'backstory': 'You are skilled at web scraping, API interactions, and data extraction from online sources.',
                'tools': [self.search_tool],
                'code_template': '''
import requests
from bs4 import BeautifulSoup
import json
import time

class WebAgent:
    def __init__(self):
        self.session = requests.Session()

    def execute_task(self):
        """Generated based on user prompt"""
        # Web scraping/API logic here
        print(f"Web agent executing: {prompt}")
        return "Web task completed"

if __name__ == "__main__":
    agent = WebAgent()
    result = agent.execute_task()
    print(result)
'''
            },
            'communication': {
                'role': 'Communication Specialist',
                'goal': f'Handle communication tasks: {prompt}',
                'backstory': 'You excel at sending notifications, emails, and managing communication workflows.',
                'tools': [],
                'code_template': '''
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

class CommunicationAgent:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.port = 587

    def execute_task(self):
        """Generated based on user prompt"""
        # Communication logic here
        print(f"Communication agent executing: {prompt}")
        return "Communication task completed"

if __name__ == "__main__":
    agent = CommunicationAgent()
    result = agent.execute_task()
    print(result)
'''
            },
            'general': {
                'role': 'General Purpose Assistant',
                'goal': f'Execute general tasks: {prompt}',
                'backstory': 'You are a versatile assistant capable of handling various computational and analytical tasks.',
                'tools': [self.search_tool],
                'code_template': '''
import json
import time
from datetime import datetime

class GeneralAgent:
    def __init__(self):
        self.start_time = datetime.now()

    def execute_task(self):
        """Generated based on user prompt"""
        # General purpose logic here
        print(f"General agent executing: {prompt}")
        return "General task completed"

if __name__ == "__main__":
    agent = GeneralAgent()
    result = agent.execute_task()
    print(result)
'''
            }
        }

        return configs.get(category, configs['general'])

    def generate_agent(self, prompt):
        """Generate a CrewAI agent based on user prompt"""
        category = self.parse_prompt(prompt)
        config = self.create_agent_config(prompt, category)

        # Create CrewAI agent
        agent = Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=config['tools'],
            verbose=True,
            allow_delegation=False
        )

        # Create a task for the agent
        task = Task(
            description=prompt,
            expected_output="A detailed response addressing the user's request",
            agent=agent
        )

        # Generate executable code
        code = config['code_template'].replace('{prompt}', prompt)

        return {
            'agent': agent,
            'task': task,
            'code': code,
            'category': category
        }


# Initialize the agent creator
agent_creator = AgentCreator()


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'AI Agent Creator API with CrewAI is running!'})


@app.route('/create-agent', methods=['POST'])
def create_agent():
    try:
        data = request.json
        prompt = data.get('prompt', '')

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Generate agent using CrewAI
        agent_data = agent_creator.generate_agent(prompt)
        agent_id = str(uuid.uuid4())

        # Save agent configuration and code
        agent_dir = f"generated_agents/{agent_id}"
        os.makedirs(agent_dir, exist_ok=True)

        # Save the executable code
        with open(f"{agent_dir}/agent.py", 'w') as f:
            f.write(agent_data['code'])

        # Save agent metadata
        metadata = {
            'id': agent_id,
            'prompt': prompt,
            'category': agent_data['category'],
            'role': agent_data['agent'].role,
            'goal': agent_data['agent'].goal,
            'created_at': str(uuid.uuid1().time),
            'status': 'created'
        }

        with open(f"{agent_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return jsonify({
            'agent_id': agent_id,
            'status': 'created',
            'category': agent_data['category'],
            'role': agent_data['agent'].role,
            'goal': agent_data['agent'].goal,
            'code': agent_data['code']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/agents', methods=['GET'])
def list_agents():
    try:
        agents_dir = "generated_agents"
        if not os.path.exists(agents_dir):
            return jsonify([])

        agents = []
        for agent_id in os.listdir(agents_dir):
            metadata_path = f"{agents_dir}/{agent_id}/metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    agents.append(metadata)

        return jsonify(agents)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/execute-agent/<agent_id>', methods=['POST'])
def execute_agent(agent_id):
    """Execute an agent's code (for software agents only)"""
    try:
        agent_dir = f"generated_agents/{agent_id}"
        if not os.path.exists(agent_dir):
            return jsonify({'error': 'Agent not found'}), 404

        # Load metadata to check if it's safe to execute
        with open(f"{agent_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)

        if metadata['category'] == 'hardware':
            return jsonify({
                'result': 'Hardware agents should be deployed to Raspberry Pi',
                'status': 'deployment_required'
            })

        # For software agents, we can execute them locally
        import subprocess
        result = subprocess.run(
            ['python', f"{agent_dir}/agent.py"],
            capture_output=True,
            text=True,
            timeout=30
        )

        return jsonify({
            'result': result.stdout,
            'error': result.stderr if result.stderr else None,
            'status': 'executed'
        })

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Agent execution timed out'}), 408
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
