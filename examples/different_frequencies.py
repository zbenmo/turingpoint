from typing import Generator
from turingpoint.assembly import Assembly
from turingpoint.definitions import Participant
from turingpoint.utils import print_parcel


def different_frequencies():
  
  def participant1(parcel: dict) -> None:
    parcel['participant1'] = parcel.get('participant1', 0) + 1

  def participant2(parcel: dict) -> None:
    parcel['participant2'] = parcel.get('participant2', 0) + 1

  def participant3(parcel: dict) -> None:
    parcel['participant3'] = parcel.get('participant3', 0) + 1

  def participants() -> Generator[Participant, None, None]:
    for i in range(20):
      if i % 1 == 0:
        yield participant1
      if i % 2 == 0:
        yield participant2
      if i % 3 == 0:
        yield participant3
      yield print_parcel

  assembly = Assembly(participants)
  parcel = assembly.launch()

  print(f'{parcel=}')


if __name__ == "__main__":
  different_frequencies()