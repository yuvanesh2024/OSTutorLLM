"""Gradio browser interface. Backend objects are supplied by the notebook."""
import gradio as gr
from core import TOPICS, schedule

def build_ui(tutor):
    def ask(q, mode):
        try:
            return tutor.ask(q, mode)
        except Exception as e:
            return f'Error: {e}', '', 'Check notebook output.'
    def new(student, topic):
        try:
            qid, question, options, level = tutor.new_quiz(student, topic)
            return qid, f'**{level.title()}**\n\n{question}', gr.update(choices=options, value=None), ''
        except Exception as e:
            return '', str(e), gr.update(choices=[], value=None), ''
    def grade(qid, student, answer):
        try:
            return tutor.grade(qid, student, answer)
        except Exception as e:
            return str(e)
    def calc(text, algorithm, quantum):
        import pandas as pd
        import matplotlib.pyplot as plt
        try:
            table, timeline = schedule(text, algorithm, quantum)
            frame = pd.DataFrame(table)
            fig, ax = plt.subplots(figsize=(11, 3.5))
            labels = list(dict.fromkeys(x['pid'] for x in timeline))
            colors = plt.get_cmap('tab20')
            for seg in timeline:
                pos = labels.index(seg['pid'])
                ax.barh(pos, seg['end']-seg['start'], left=seg['start'], color=colors(pos%20), edgecolor='black')
                ax.text((seg['start']+seg['end'])/2, pos, f'{seg["start"]}-{seg["end"]}', ha='center', va='center', fontsize=8)
            ax.set_yticks(range(len(labels)), labels); ax.set_xlabel('Time units')
            ax.set_title(algorithm); fig.tight_layout(); plt.close(fig)
            note = (f'Average waiting: {frame.Waiting.mean():.2f}; average turnaround: {frame.Turnaround.mean():.2f}. '
                    'TAT = completion - arrival; waiting = TAT - burst. '
                    'Single CPU, no I/O, zero context-switch cost. SJF is non-preemptive. '
                    'RR queues arrivals at a slice boundary before requeueing the running process.')
            return frame, fig, note
        except Exception as e:
            return None, None, str(e)
    with gr.Blocks(title='OSTutorLLM') as demo:
        gr.Markdown('# OSTutorLLM\nCourse-grounded explanations, adaptive practice, and a CPU scheduling lab.')
        with gr.Tab('Ask your notes'):
            question = gr.Textbox(label='Self-contained OS question', lines=3, max_lines=6)
            mode = gr.Dropdown(['Explain simply', 'Exam answer', 'Viva practice', 'Shell lab'], value='Explain simply', label='Answer style')
            button = gr.Button('Ask', variant='primary')
            answer = gr.Markdown()
            status = gr.Textbox(label='Timing / retrieval status')
            with gr.Accordion('Inspect retrieved evidence', open=False):
                evidence = gr.Markdown()
            button.click(ask, [question, mode], [answer, evidence, status])
            gr.Markdown('Each request is independent; repeat the topic in follow-up questions. Similarity is not confidence. Generated code is displayed only, never executed.')
        with gr.Tab('Adaptive practice'):
            student = gr.Textbox(value='student1', label='Student pseudonym (not a login)')
            topic = gr.Dropdown(list(TOPICS), value='CPU scheduling', label='Topic')
            qid = gr.State('')
            new_button = gr.Button('New question')
            quiz_text = gr.Markdown()
            choices = gr.Radio([], label='Your answer')
            submit = gr.Button('Submit answer')
            feedback = gr.Markdown()
            new_button.click(new, [student, topic], [qid, quiz_text, choices, feedback])
            submit.click(grade, [qid, student, choices], feedback)
            progress_button = gr.Button('Show my progress')
            progress = gr.Markdown()
            progress_button.click(tutor.progress, student, progress)
            gr.Markdown('The latest five attempts control difficulty after at least three answers: <50% easy, 50–79% medium, ≥80% hard. LLM-written questions still need human quality review.')
        with gr.Tab('Verify CPU scheduling'):
            processes = gr.Textbox(value='P1,0,5\nP2,1,3\nP3,2,1', label='PID,arrival,burst — one per line, no header', lines=5)
            alg = gr.Dropdown(['FCFS', 'SJF (non-preemptive)', 'Round Robin'], value='FCFS', label='Algorithm')
            quantum = gr.Number(value=2, precision=0, minimum=1, label='Round Robin time quantum')
            run = gr.Button('Calculate and draw Gantt chart')
            table = gr.Dataframe(label='Python-computed results', interactive=False)
            plot = gr.Plot()
            notes = gr.Markdown()
            run.click(calc, [processes, alg, quantum], [table, plot, notes])
            explain = gr.Button('Explain this computed schedule using my notes')
            explanation = gr.Markdown()
            def explain_schedule(text, algorithm, quantum):
                try:
                    values, _ = schedule(text, algorithm, quantum)
                    q = f'Explain {algorithm} CPU scheduling and the waiting/turnaround formulas. '
                    base, _, _ = tutor.ask(q)
                    return base + '\n\n**Exact results from Python:**\n\n' + '\n'.join(
                        f'- {r["PID"]}: completion={r["Completion"]}, turnaround={r["Turnaround"]}, waiting={r["Waiting"]}' for r in values)
                except Exception as e:
                    return str(e)
            explain.click(explain_schedule, [processes, alg, quantum], explanation)
    return demo.queue(default_concurrency_limit=1, max_size=12)
