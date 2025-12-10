from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, List, Union


@dataclass(frozen=True)
class CreaBuilding:
    """ビルのエリア情報"""

    AREA_UNITS: ClassVar[List[str]] = [
        "Area1",
        "Area2",
        "Area3",
        "Area4",
        "MeetingRoom",
        "BreakRoom",
    ]
    Unit_dict: ClassVar[Dict[str, Union[str, list]]] = dict(
        Area1={
            "idu": [
                "E-9南1",
                "E-10南2",
                "E-13北1",
                "E-11南3",
                "E-14北2",
                "E-12南4",
                "E-15北3",
                "E-16北4",
            ],
            "odu": ["49-8", "49-7", "49-6", "49-4", "49-4", "49-3", "49-2", "49-1"],
        },
        Area2={
            "idu": ["D-2北1", "D-4北2", "D-5南1", "D-7南2", "D-6北1", "D-8北2"],
            "odu": ["44-3", "44-1", "43-4", "43-3", "43-2", "43-1"],
        },
        Area3={
            "idu": ["E-17", "A-26"],
            "odu": ["49-9", "49-9"],
        },
        Area4={
            "idu": ["F-20", "F-19", "F-18"],
            "odu": ["44-4", "44-4", "44-2"],
        },
        MeetingRoom={
            "idu": ["G-24", "G-23"],
            "odu": ["44-8", "44-8"],
        },
        BreakRoom={
            "idu": ["G-21", "G-22"],
            "odu": ["44-6", "44-6"],
        },
    )
    weather_colimns: ClassVar[List[str]] = [
        "Outdoor Temp.",
        "Outdoor Humidity",
        "Solar Radiation",
    ]
    time_columns: ClassVar[List[str]] = ["hour", "month", "weekday", "is_weekend"]

    """ 以下の関数は補助関数である。 """

    @property
    def common_columns(self):
        return self.weather_colimns + self.time_columns

    @classmethod
    def get_columns_by_area_units(
        cls,
        columns: Iterable[str],
        area: str,
    ) -> List[str]:
        if area not in cls.AREA_UNITS:
            raise ValueError(
                f"Unknown area: {area}. 定義されているエリア: {list(cls.AREA_UNITS.keys())}"
            )

        idu_names = set(cls.Unit_dict[area]["idu"])
        odu_names = set(cls.Unit_dict[area]["odu"])

        picked: List[str] = []
        for col in columns:
            unit_name = col.split("__")[-1]  # '__' の最後の部分をユニット名として扱う
            if unit_name in idu_names or unit_name in odu_names:
                picked.append(col)

        return picked

    @classmethod
    def pick_cols(cls, columns: list, prefix: str) -> list[str]:
        return [c for c in columns if c.startswith(prefix)]
