@REM  @echo off
:: .\run_all.bat
:: python main.py
:: run_experiments\###.bat

@REM  run_experiments\AIDS.bat
@REM  run_experiments\BAS.bat
@REM  run_experiments\BBBP.bat
@REM  run_experiments\BZR.bat
@REM  run_experiments\COLLAB.bat
@REM  run_experiments\COLORS-3.bat
@REM  run_experiments\COX2.bat
@REM  run_experiments\DBLP.bat
@REM  run_experiments\ENZYMES.bat
@REM  run_experiments\Fingerprint.bat
@REM  run_experiments\MUTAG.bat

@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.1-50-0.1.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.01-50-0.1.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.1-50-0.5.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.01-50-0.5.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.1-100-0.1.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.01-100-0.1.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.1-100-0.5.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.01-100-0.5.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.1-500-0.1.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.01-500-0.1.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.1-500-0.5.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-0.01-500-0.5.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-1-50-0.1.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-1-50-0.5.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-1-100-0.1.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-1-100-0.5.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-1-500-0.1.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\TCR-hypersearch\TCR-5000-28-0.3_GCN_XPlore-1-500-0.5.jsonc

python main.py XPlore_config\GRAPH_Full\TG\TG_GCN_XPlore.jsonc
python main.py XPlore_config\GRAPH_Full\TG\TG_GCN_XPlore+.jsonc
python main.py XPlore_config\GRAPH_Full\TG\TG_GCN_XPlore++.jsonc
python main.py XPlore_config\GRAPH_Full\TG\TG_GCN_XPlore-.jsonc
python main.py XPlore_config\GRAPH_Full\TG\TG_GCN_XPlore--.jsonc
python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_XPlore.jsonc
python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_XPlore+.jsonc
python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_XPlore++.jsonc
python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_XPlore-.jsonc
python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_XPlore--.jsonc