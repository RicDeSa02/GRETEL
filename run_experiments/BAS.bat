:: run_experiments\BAS.bat
:: python main.py

@REM  python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_CF-GNNE.jsonc
@REM  python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_CF2.jsonc
python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_CLEAR.jsonc
python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_COMBINEX.jsonc
python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_D4Explainer.jsonc
python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_iRand.jsonc
python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_RSGG-CE.jsonc
python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_XPlore.jsonc
@REM python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_XPlore+.jsonc
@REM python main.py XPlore_config\GRAPH_Full\BAS\BAS_GCN_XPlore++.jsonc