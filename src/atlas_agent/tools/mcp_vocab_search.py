"""Direct MCP Vocab search tool - calls omcp_vocab MCP server directly."""
import json
import subprocess
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models import ConceptMatch


class MCPVocabSearchTool:
    """
    Search OMOP vocabulary via omcp_vocab MCP server.

    Makes direct JSON-RPC calls to the MCP server subprocess,
    bypassing LLM agent reasoning for fast, deterministic results.
    """

    def __init__(
        self,
        omcp_vocab_path: str = "/Users/k24118093/Documents/omcp_vocab"
    ):
        """
        Initialize MCP Vocab search tool.

        Args:
            omcp_vocab_path: Path to omcp_vocab directory
        """
        self.omcp_vocab_path = omcp_vocab_path
        self._process = None
        self._request_id = 0

    def _start_server(self):
        """Start the omcp_vocab MCP server subprocess."""
        if self._process is None:
            cmd = [
                "uv", "--directory", self.omcp_vocab_path,
                "run", "python", "-m", "omcp_vocab.main"
            ]
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

    def _stop_server(self):
        """Stop the MCP server subprocess."""
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None

    def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool via JSON-RPC.

        Args:
            tool_name: Name of the tool (lookup_condition, lookup_drug, lookup_observation)
            arguments: Tool arguments

        Returns:
            Tool response as dict
        """
        self._start_server()

        # Build JSON-RPC request
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        # Send request
        request_str = json.dumps(request) + "\n"
        self._process.stdin.write(request_str)
        self._process.stdin.flush()

        # Read response
        response_str = self._process.stdout.readline()
        response = json.loads(response_str)

        if "error" in response:
            raise Exception(f"MCP error: {response['error']}")

        return response.get("result", {})

    def search_concepts(
        self,
        query_text: str,
        domain_filter: Optional[str] = None,
        top_k: int = 20,
        min_similarity: float = 0.5,
        matched_entity: Optional[str] = None,
        **kwargs
    ) -> List[ConceptMatch]:
        """
        Search for concepts using omcp_vocab MCP server.

        Args:
            query_text: Text to search for
            domain_filter: Optional domain filter (Condition, Drug, Measurement, Observation)
            top_k: Maximum number of results
            min_similarity: Not used (keyword search)
            matched_entity: Entity text that matched this search

        Returns:
            List of ConceptMatch objects
        """
        try:
            # Map domain to MCP tool
            tool_name = None
            if domain_filter == "Condition":
                tool_name = "lookup_condition"
            elif domain_filter == "Drug":
                tool_name = "lookup_drug"
            elif domain_filter in ["Measurement", "Observation"]:
                tool_name = "lookup_observation"
            else:
                # Default to condition for unknown domains
                tool_name = "lookup_condition"

            # Call MCP tool
            result = self._call_mcp_tool(
                tool_name=tool_name,
                arguments={
                    "query": query_text,
                    "page_size": top_k
                }
            )

            # Parse response - omcp_vocab returns JSON in text content
            content = result.get("content", [])
            if not content:
                return []

            # First content item contains the summary + JSON
            text = content[0].get("text", "")

            # Extract JSON from text (after "Full JSON:\n")
            json_start = text.find("Full JSON:\n")
            if json_start == -1:
                return []

            json_str = text[json_start + len("Full JSON:\n"):]
            concepts = json.loads(json_str)

            # Convert to ConceptMatch objects
            matches = []
            for idx, concept in enumerate(concepts):
                # Rank-based pseudo-similarity score
                similarity = max(0.1, 1.0 - (idx * 0.05))

                # Convert timestamps to ISO date strings
                valid_start = datetime.fromtimestamp(
                    concept.get("VALID_START_DATE", 0) / 1000
                ).strftime("%Y-%m-%d") if concept.get("VALID_START_DATE") else "1970-01-01"

                valid_end = datetime.fromtimestamp(
                    concept.get("VALID_END_DATE", 0) / 1000
                ).strftime("%Y-%m-%d") if concept.get("VALID_END_DATE") else "2099-12-31"

                match = ConceptMatch(
                    concept_id=int(concept["CONCEPT_ID"]),
                    concept_name=concept["CONCEPT_NAME"],
                    concept_code=concept.get("CONCEPT_CODE", ""),
                    vocabulary_id=concept.get("VOCABULARY_ID", ""),
                    domain_id=concept.get("DOMAIN_ID", domain_filter or ""),
                    concept_class_id=concept.get("CONCEPT_CLASS_ID", ""),
                    standard_concept=concept.get("STANDARD_CONCEPT", ""),
                    valid_start_date=valid_start,
                    valid_end_date=valid_end,
                    invalid_reason=concept.get("INVALID_REASON"),
                    similarity_score=similarity,
                    matched_entity=matched_entity or query_text
                )
                matches.append(match)

            return matches

        except Exception as e:
            print(f"⚠️  MCP vocab search failed: {e}")
            return []

    def search_by_code(self, code: str, vocabulary_id: str) -> Optional[ConceptMatch]:
        """
        Search for a concept by its code.

        Args:
            code: Concept code
            vocabulary_id: Vocabulary ID

        Returns:
            ConceptMatch if found, None otherwise
        """
        # Search and filter for exact match
        results = self.search_concepts(query_text=code, top_k=10)

        for result in results:
            if result.concept_code == code and result.vocabulary_id == vocabulary_id:
                return result

        return None

    def __del__(self):
        """Cleanup subprocess on deletion."""
        self._stop_server()
