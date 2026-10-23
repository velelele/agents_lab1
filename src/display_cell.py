import sys
import json
from typing import Any, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

def render_cell(title: str, content: Any, border_style: str = "bold cyan", subtitle: Optional[str] = None) -> None:
    is_jupyter = False
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            is_jupyter = True
    except Exception:
        is_jupyter = False

    if is_jupyter:
        try:
            from IPython.display import display, HTML
            formatted_html = f"""
            <div style="border: 2px solid #5846c7; background: #f5f6fb; padding: 14px; border-radius: 12px; margin: 10px 0; font-family: monospace;">
                <div style="font-weight: bold; color: #5846c7; font-size: 14px; margin-bottom: 6px;">[CELL] {title}</div>
                <pre style="margin: 0; white-space: pre-wrap; font-size: 13px; color: #20243a;">{str(content)}</pre>
            </div>
            """
            display(HTML(formatted_html))
            return
        except Exception:
            pass

    if isinstance(content, dict):
        table = Table(show_header=True, header_style="bold magenta", expand=True, box=None)
        table.add_column("Параметр / Факт", style="bold yellow", width=24)
        table.add_column("Значение", style="white")
        for key, value in content.items():
            table.add_row(str(key), str(value))
        panel = Panel(table, title=f"[bold cyan][ CELL: {title} ][/bold cyan]", subtitle=subtitle, border_style=border_style, padding=(1, 2))
    else:
        text_obj = Text(str(content), style="white")
        panel = Panel(text_obj, title=f"[bold cyan][ CELL: {title} ][/bold cyan]", subtitle=subtitle, border_style=border_style, padding=(1, 2))

    console.print()
    console.print(panel)
    console.print()
