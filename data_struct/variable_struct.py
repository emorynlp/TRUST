import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import utils.file as uf


def get_section(query_file: str) -> str:
    fname = Path(query_file).stem
    sections = re.findall(r"[A-Z]+\d?", fname)
    if sections:
        return sections[0]
    else:
        raise ValueError(f"Invalid query file name: {query_file}. No section found.")


def get_level(query: str) -> int:
    return query[:10].count("-")


def check_recurrent(query: str) -> bool:
    return False if query[:10].count("*") == 0 else True


def get_condition(query: str) -> str | None:
    if "[" in query and "]" in query:
        return query[query.find("[") + 1 : query.find("]")]
    else:
        return None


def get_question(query: str) -> str:
    if "[" in query and "]" in query:
        question = query.split("]")[1]
    else:
        if not check_recurrent(query):
            question = query[get_level(query) :]
        else:
            question = query[get_level(query) + 1 :]
    return question.strip()


def add_query(query: list, q_idx_start: int) -> list:
    nodes = format_query(query, q_idx_start)
    return [Query(*node) for node in nodes]
    # for i, node in enumerate(nodes):


def format_query(questions: list, q_idx_start: int) -> list:
    nodes = []
    id_to_node = {}
    hierarchy = {k: -1 for k in range(6)}
    for i, q in enumerate(questions):
        level = get_level(q)
        q_idx = q_idx_start + i
        question = get_question(q)
        parent_idx = hierarchy[level - 1] if level > 0 else -1
        hierarchy[level] = i
        recurrent = check_recurrent(q)
        condition = get_condition(q)

        node = (q_idx, question, level, recurrent, condition, parent_idx, [])
        nodes.append(node)
        id_to_node[i] = node

        if parent_idx != -1:
            id_to_node[parent_idx][-1].append(i)
    return nodes


def add_prerequisites(prerequisites: str) -> list[str]:
    if prerequisites == "":
        return []
    elif prerequisites.startswith("IA") or prerequisites.startswith("calculation"):
        return [prerequisites]
    else:
        return prerequisites.split(", ")


def format_meta(meta_dict: dict) -> "VariableMeta":
    var_type = meta_dict.get("field_type", "unknown")
    template = meta_dict.get("template", {})
    patterns = {
        k: v for k, v in meta_dict.items() if k not in ["field_type", "template"]
    }

    return VariableMeta(
        var_type=var_type,
        patterns=patterns,
        template=template,
    )


def format_vars(query_file: str, meta_file: str | None) -> list:
    data = uf.File(query_file).load()
    meta_data = uf.File(meta_file).load() if meta_file else None
    vars = []
    q_idx_start = 0
    for i, (var_id, var_data) in enumerate(data.items()):
        if isinstance(var_data, dict):
            queries = (
                add_query(var_data["questions"], q_idx_start=q_idx_start)
                if var_data["questions"]
                else []
            )
            var = Variable(
                vid=var_id,
                var_idx=i,
                queries=queries,
                metadata=format_meta(meta_data.get(var_id))
                if meta_data and var_id in meta_data
                else None,
                prerequisites=add_prerequisites(var_data["prerequisites"]),
            )
            q_idx_start += len(queries)
        elif isinstance(var_data, list):  # questions
            queries = add_query(var_data, q_idx_start=q_idx_start)
            var = Variable(vid=var_id, queries=queries, var_idx=i)
            q_idx_start += len(queries)
        else:
            # 1. empty str: no question; 2. str: variable(s); 3. str: interviewer assessment
            var = Variable(
                vid=var_id, prerequisites=add_prerequisites(var_data), var_idx=i
            )
        vars.append(var)
    return vars


@dataclass
class Query:
    qid: int
    question: str
    level: int
    recurrent: bool
    condition: str | None
    parent_idx: int
    children: list[int] = field(default_factory=list)


@dataclass
class VariableMeta:
    var_type: Literal["scale", "measure", "category", "notes", "rule", "IA"]
    patterns: dict = field(default_factory=dict)
    template: dict = field(default_factory=dict)


@dataclass
class ScaleMeta(VariableMeta):
    range: dict | None = None


@dataclass
class Variable:
    vid: str
    var_idx: int
    metadata: VariableMeta | None = None
    prerequisites: list[str] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)

    def __getitem__(self, idx: int) -> Query:
        return self.queries[idx]

    def __len__(self) -> int:
        return len(self.queries)

    def get_queries_by_level(
        self, level: int = 0, start_qid: int | None = None
    ) -> list[Query]:
        if start_qid:
            return [q for q in self.queries if q.level == level and q.qid > start_qid]
        return [q for q in self.queries if q.level == level]

    def get_query_by_qid(self, qid: int) -> Query | None:
        query = [q for q in self.queries if q.qid == qid]
        return query[0] if query else None

    def get_sibling_queries(self, query: Query, filter_level: int = 1) -> list[Query]:
        siblings = self.get_queries_by_level(level=query.level, start_qid=query.qid)
        if siblings and query.level > filter_level:
            siblings = [s for s in siblings if s.parent_idx == query.parent_idx]
        return siblings

    def get_parent_queries(self, query: Query) -> list[Query]:
        queries = []
        # parent = self.get_query_by_qid(query.parent)
        parent = self.queries[query.parent_idx]
        if parent and parent.level > 0:
            queries.extend(self.get_sibling_queries(parent))
            queries.extend(self.get_parent_queries(parent))
        return queries


@dataclass
class SectionVars:
    query_file: str = field(repr=False)
    meta_file: str | None = field(repr=False, default=None)
    section: str = field(init=False)
    variables: list[Variable] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.section = get_section(self.query_file)
        self.variables = format_vars(self.query_file, self.meta_file)

    def __getitem__(self, key) -> Variable:
        return self.variables[key]

    def __len__(self) -> int:
        return len(self.variables)

    def get_var_by_qid(self, qid: int) -> tuple | None:
        question = [(v, q) for v in self.variables for q in v.queries if q.qid == qid]
        return question[0] if question else None

    def var_index_by_id(self, varname: str) -> int | None:
        return [v.vid for v in self.variables].index(varname)

    def get_core_questions(self) -> list:
        return [q for v in self.variables for q in v.queries if q.level == 0]

    def get_queries_in_next_var(self, var_idx: int) -> list:
        queries = []

        if var_idx < len(self.variables) - 1:
            queries.extend(self.variables[var_idx + 1].get_queries_by_level())
            if not queries:
                queries.extend(
                    self.variables[var_idx + 1].get_queries_by_level(level=1)
                )

        if not queries and var_idx < len(self.variables) - 2:
            return self.get_queries_in_next_var(var_idx + 1)

        return queries


if __name__ == "__main__":
    # Example usage
    query_file = "CAPS/agent_vars/CAPS5.json"
    meta_file = "CAPS/agent_templates/CAPS5_var_template.json"
    section_vars = SectionVars(query_file=query_file, meta_file=meta_file)
    print(section_vars.section)
    # print(section_vars.variables)
