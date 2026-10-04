:: run_experiments\ENZYMES.bat
:: python main.py

python main.py XPlore_config\GRAPH\ENZYMES\ENZYMES_GCN_CF-GNNE.jsonc
python main.py XPlore_config\GRAPH\ENZYMES\ENZYMES_GCN_CF2.jsonc
python main.py XPlore_config\GRAPH\ENZYMES\ENZYMES_GCN_CLEAR.jsonc
python main.py XPlore_config\GRAPH\ENZYMES\ENZYMES_GCN_D4Explainer.jsonc
python main.py XPlore_config\GRAPH\ENZYMES\ENZYMES_GCN_iRand.jsonc
python main.py XPlore_config\GRAPH\ENZYMES\ENZYMES_GCN_RSGG-CE.jsonc
@REM python main.py XPlore_config\GRAPH\ENZYMES\ENZYMES_GCN_XPlore.jsonc
@REM python main.py XPlore_config\GRAPH\ENZYMES\ENZYMES_GCN_XPlore+.jsonc
@REM python main.py XPlore_config\GRAPH\ENZYMES\ENZYMES_GCN_XPlore++.jsonc