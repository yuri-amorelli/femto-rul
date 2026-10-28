from src import config
from src.pipeline import prepare_bearing

for n in ["Bearing1_3", "Bearing2_3", "Bearing3_3"]:
    _, yt = prepare_bearing(config.DATA_ROOT / "Test_set" / n, label_mode="capped", cache=False)
    _, yf = prepare_bearing(config.DATA_ROOT / "Full_Test_Set" / n, label_mode="capped", cache=False)
    true_rul = (len(yf) - len(yt)) * 10
    print(f"{n}: troncato {len(yt)} snapshot, ultima etichetta del codice = {yt[-1]:.0f} s | "
          f"completo {len(yf)} snapshot -> RUL reale al troncamento = {true_rul} s")