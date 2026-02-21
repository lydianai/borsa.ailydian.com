"""
AILYDIAN Backend Client
Main client class for accessing backend services
"""

import os
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
from .exceptions import (
    AILYDIANError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    ProjectNotFoundError,
    DashboardAccessError
)


class LLMService:
    """LLM Router service wrapper"""

    def __init__(self, client):
        self.client = client

    def route(self, prompt: str, context: Optional[Dict[str, Any]] = None,
             model: Optional[str] = None) -> Dict[str, Any]:
        """Route a prompt through the LLM Router"""
        return self.client._request('POST', '/llm/route', {
            'prompt': prompt,
            'context': context or {},
            'model': model
        })

    def get_stats(self) -> Dict[str, Any]:
        """Get LLM routing statistics"""
        return self.client._request('GET', '/llm/stats')


class VectorStoreService:
    """Vector Store service wrapper"""

    def __init__(self, client):
        self.client = client

    def add_documents(self, documents: List[str], collection: str = "default",
                     metadata: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Add documents to vector store"""
        return self.client._request('POST', '/vector/add', {
            'documents': documents,
            'collection': collection,
            'metadata': metadata
        })

    def search(self, query: str, collection: str = "default", k: int = 5,
              filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search in vector store"""
        result = self.client._request('POST', '/vector/search', {
            'query': query,
            'collection': collection,
            'k': k,
            'filter': filter
        })
        return result.get('results', [])

    def delete_collection(self, collection: str) -> Dict[str, Any]:
        """Delete a collection"""
        return self.client._request('DELETE', f'/vector/collection/{collection}')


class KnowledgeGraphService:
    """Knowledge Graph service wrapper"""

    def __init__(self, client):
        self.client = client

    def add_entity(self, entity_type: str, name: str,
                  properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add an entity to the knowledge graph"""
        return self.client._request('POST', '/graph/entity', {
            'type': entity_type,
            'name': name,
            'properties': properties or {}
        })

    def add_relation(self, source: str, relation: str, target: str,
                    properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add a relation between entities"""
        return self.client._request('POST', '/graph/relation', {
            'source': source,
            'relation': relation,
            'target': target,
            'properties': properties or {}
        })

    def query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute a Cypher-like query"""
        result = self.client._request('POST', '/graph/query', {
            'query': cypher_query
        })
        return result.get('results', [])


class AgentService:
    """Agent Ecosystem service wrapper"""

    def __init__(self, client):
        self.client = client

    def create(self, agent_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new agent"""
        return self.client._request('POST', '/agents/create', {
            'agent_type': agent_type,
            'config': config or {}
        }, resource_type='agents')

    def list(self) -> List[Dict[str, Any]]:
        """List all agents for this project"""
        result = self.client._request('GET', '/agents/list')
        return result.get('agents', [])

    def execute(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with an agent"""
        return self.client._request('POST', f'/agents/{agent_id}/execute', {
            'task': task
        })

    def get_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent status"""
        return self.client._request('GET', f'/agents/{agent_id}/status')

    def stop(self, agent_id: str) -> Dict[str, Any]:
        """Stop an agent"""
        return self.client._request('POST', f'/agents/{agent_id}/stop')


class EcosystemAgentService:
    """Ecosystem Agents service wrapper (Data Analysis, Research, etc.)"""

    def __init__(self, client):
        self.client = client

    def create(self, agent_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create an ecosystem agent"""
        return self.client._request('POST', '/ecosystem-agents/create', {
            'agent_type': agent_type,
            'config': config or {}
        }, resource_type='agents')

    def list(self) -> List[Dict[str, Any]]:
        """List all ecosystem agents"""
        result = self.client._request('GET', '/ecosystem-agents/list')
        return result.get('agents', [])

    def types(self) -> List[Dict[str, Any]]:
        """Get available ecosystem agent types"""
        result = self.client._request('GET', '/ecosystem-agents/types')
        return result.get('agent_types', [])

    def execute(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with ecosystem agent"""
        return self.client._request('POST', f'/ecosystem-agents/{agent_id}/execute', {
            'task': task
        })

    def get_metrics(self) -> Dict[str, Any]:
        """Get ecosystem agents metrics"""
        return self.client._request('GET', '/ecosystem-agents/metrics')


class AILYDIANBackend:
    """Main AILYDIAN Backend client"""

    def __init__(self, base_url: str = "http://localhost:8888",
                 project_id: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize AILYDIAN Backend client

        Args:
            base_url: Backend server URL
            project_id: Project identifier (auto-detected if not provided)
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.project_id = project_id or self._detect_project_id()
        self.api_key = api_key
        self.session = requests.Session()

        # Set up headers
        self.session.headers.update({
            'X-Project-ID': self.project_id,
            'Content-Type': 'application/json'
        })

        if self.api_key:
            self.session.headers['X-API-Key'] = self.api_key

        # Initialize service wrappers
        self.llm = LLMService(self)
        self.vector = VectorStoreService(self)
        self.graph = KnowledgeGraphService(self)
        self.agents = AgentService(self)
        self.ecosystem_agents = EcosystemAgentService(self)

    def _detect_project_id(self) -> str:
        """Auto-detect project ID from current working directory"""
        cwd = Path.cwd()

        # Check if we're in a known project directory
        project_markers = [
            'package.json',
            'requirements.txt',
            'setup.py',
            '.git'
        ]

        # Try to detect from directory name (e.g., travel.ailydian.com)
        if 'ailydian.com' in cwd.name:
            return cwd.name

        # Try parent directory
        if 'ailydian.com' in cwd.parent.name:
            return cwd.parent.name

        # Fallback: use directory name
        return cwd.name

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None,
                resource_type: Optional[str] = None) -> Dict[str, Any]:
        """Make HTTP request to backend"""

        # Construct full URL
        url = f"{self.base_url}/api/project/{self.project_id}{endpoint}"

        try:
            if method == 'GET':
                response = self.session.get(url, timeout=30)
            elif method == 'POST':
                response = self.session.post(url, json=data, timeout=30)
            elif method == 'DELETE':
                response = self.session.delete(url, timeout=30)
            else:
                raise AILYDIANError(f"Unsupported HTTP method: {method}")

            # Handle response
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                error_data = response.json()
                code = error_data.get('code', '')

                if code == 'PROJECT_NOT_REGISTERED':
                    raise ProjectNotFoundError(error_data.get('error'))
                elif code == 'DASHBOARD_ACCESS_FORBIDDEN':
                    raise DashboardAccessError(error_data.get('error'))
                else:
                    raise AuthenticationError(error_data.get('error'))

            elif response.status_code == 429:
                error_data = response.json()
                code = error_data.get('code', '')

                if code == 'RATE_LIMIT_EXCEEDED':
                    raise RateLimitError(
                        error_data.get('error'),
                        current_usage=error_data.get('current_usage'),
                        limit=error_data.get('limit')
                    )
                elif code == 'QUOTA_EXCEEDED':
                    raise QuotaExceededError(
                        error_data.get('error'),
                        resource_type=resource_type,
                        current_usage=error_data.get('current_usage'),
                        quota=error_data.get('quota')
                    )
                else:
                    raise AILYDIANError(error_data.get('error'))

            else:
                try:
                    error_data = response.json()
                    raise AILYDIANError(f"Request failed: {error_data.get('error', 'Unknown error')}")
                except:
                    raise AILYDIANError(f"Request failed with status {response.status_code}")

        except requests.exceptions.ConnectionError:
            raise AILYDIANError(f"Could not connect to AILYDIAN backend at {self.base_url}")
        except requests.exceptions.Timeout:
            raise AILYDIANError("Request timed out")
        except requests.exceptions.RequestException as e:
            raise AILYDIANError(f"Request failed: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """Get project status and usage statistics"""
        return self._request('GET', '/status')

    def analyze(self, data: Any, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze data using AILYDIAN backend"""
        return self._request('POST', '/analyze', {
            'data': data,
            'analysis_type': analysis_type
        })

    def research(self, query: str, sources: Optional[List[str]] = None,
                depth: str = "medium") -> Dict[str, Any]:
        """Perform deep research using AILYDIAN backend"""
        return self._request('POST', '/research', {
            'query': query,
            'sources': sources or ['web', 'academic'],
            'depth': depth
        })
