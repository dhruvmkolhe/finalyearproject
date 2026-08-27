import json
import os
import io
import base64
import contextlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def execute_notebook():
    notebook_filename = "PredictIQ_Presentation_Notebook.ipynb"
    output_filename = "PredictIQ_Presentation_Notebook.ipynb"
    
    with open(notebook_filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Globals for notebook execution context
    exec_globals = {}
    
    print("Executing notebook cells to pre-render outputs...")
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            code = "".join(cell['source'])
            print(f"Executing code cell {idx+1}/{len(nb['cells'])}...")
            
            stdout_io = io.StringIO()
            cell_outputs = []
            
            # Define custom display handler for pandas DataFrame in script mode
            def custom_display(obj):
                if isinstance(obj, pd.DataFrame):
                    stdout_io.write(obj.to_string() + "\n\n")
                elif hasattr(obj, '_repr_html_'):
                    stdout_io.write(obj.to_string() + "\n\n")
                else:
                    stdout_io.write(str(obj) + "\n\n")
            
            exec_globals['display'] = custom_display

            try:
                plt.clf()
                plt.close('all')
                with contextlib.redirect_stdout(stdout_io):
                    exec(code, exec_globals)
                
                stdout_text = stdout_io.getvalue()
                if stdout_text.strip():
                    cell_outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [line + "\n" for line in stdout_text.splitlines()]
                    })
                
                # Check if a matplotlib figure was generated
                if plt.get_fignums():
                    fig = plt.gcf()
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
                    img_buf.seek(0)
                    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
                    
                    cell_outputs.append({
                        "data": {
                            "image/png": img_base64
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    })
                    plt.close('all')

            except Exception as e:
                print(f"Error executing cell {idx+1}: {e}")
                cell_outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": [f"Execution Note: {str(e)}\n"]
                })
            
            cell['outputs'] = cell_outputs
            cell['execution_count'] = (idx // 2) + 1

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
        
    print(f"Successfully executed and pre-rendered all cell outputs in {output_filename}")

if __name__ == "__main__":
    execute_notebook()
