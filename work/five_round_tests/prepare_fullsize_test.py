from pathlib import Path
root=Path(__file__).resolve().parents[2]
source=(root/'work/test_research_fullsize.py').read_text(encoding='utf-8')
source=source.replace('Path(__file__).resolve().parent.parent','Path(__file__).resolve().parents[2]')
source=source.replace('outputs/fovea_research_kaggle.py','outputs/fovea_optimized_5rounds.py')
source=source.replace('work/research_fullsize_check.json','work/five_round_tests/final_fullsize_report.json')
(root/'work/five_round_tests/test_final_fullsize.py').write_text(source,encoding='utf-8')
