"""Turns a hypercube point (an int bitmask) into the text of a resume "Skills" section.

Every section lists the same baseline skills. Bit i of the point says whether
stretch skill i is appended to its category's list. The data lives in data/skills.json.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

DEFAULT_SKILLS_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.json"


class SkillCategory(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    baseline_skills: list[str] = Field(min_length=1)


class StretchSkill(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    category_name: str


class SkillsCatalog(BaseModel):
    """Stretch skills are ordered from most to least relevant to the job posting."""

    model_config = ConfigDict(frozen=True)

    section_heading: str = "Skills"
    categories: list[SkillCategory] = Field(min_length=1)
    stretch_skills: list[StretchSkill]

    @classmethod
    def load_from_file(
        cls, path: Path = DEFAULT_SKILLS_PATH, dimension: NonNegativeInt = 35
    ) -> SkillsCatalog:
        catalog = cls.model_validate_json(path.read_text(encoding="utf-8"))
        return catalog.model_copy(update={"stretch_skills": catalog.stretch_skills[:dimension]})

    def render_skills_section(self, point: int) -> str:
        included_skills = [skill for index, skill in enumerate(self.stretch_skills) if point >> index & 1]
        lines = [self.section_heading]
        for category in self.categories:
            extra_skill_names = [skill.name for skill in included_skills if skill.category_name == category.name]
            lines.append(f"{category.name}: {', '.join(category.baseline_skills + extra_skill_names)}")
        return "\n".join(lines)
