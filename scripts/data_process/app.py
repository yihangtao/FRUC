from flask import Flask, render_template, request, jsonify
import os
import glob
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app = Flask(__name__, template_folder=os.path.join(PROJECT_ROOT, 'templates'))

DATA_ROOTS = {'V2X-Real': os.environ.get('V2XREAL_ROOT', os.path.join(PROJECT_ROOT, 'data', 'v2xreal'))}

def get_scenes(dataset, split):
    # dataset might have spaces if '+' was decoded incorrectly, but we'll use encodeURIComponent in JS
    if dataset not in DATA_ROOTS:
        dataset = dataset.replace(' ', '+') # Fallback
    base_dir = os.path.join(DATA_ROOTS[dataset], split)
    if not os.path.exists(base_dir):
        return []

    dirs = os.listdir(base_dir)
    scenes = set()
    for d in dirs:
        if d.endswith('_1') or d.endswith('_2'):
            scenes.add(d[:-2])
    return sorted(list(scenes))

def get_frames(dataset, split, scene):
    if dataset not in DATA_ROOTS:
        dataset = dataset.replace(' ', '+')
    base_dir = os.path.join(DATA_ROOTS[dataset], split)
    agent1_dir = os.path.join(base_dir, f"{scene}_1", "images")
    agent2_dir = os.path.join(base_dir, f"{scene}_2", "images")

    if not os.path.exists(agent1_dir) or not os.path.exists(agent2_dir):
        return []

    imgs1 = glob.glob(os.path.join(agent1_dir, "*_0.*"))
    frames1 = set([os.path.basename(f).split('_')[0] for f in imgs1])

    imgs2 = glob.glob(os.path.join(agent2_dir, "*_0.*"))
    frames2 = set([os.path.basename(f).split('_')[0] for f in imgs2])

    frames = sorted(list(frames1.intersection(frames2)))

    # Check annotations from agent1's context.json
    anno_file = os.path.join(base_dir, f"{scene}_1", "context.json")
    annotated_frames = []
    if os.path.exists(anno_file):
        with open(anno_file, 'r') as f:
            try:
                all_anno = json.load(f)
                annotated_frames = list(all_anno.keys())
            except:
                pass

    return {'frames': frames, 'annotated': annotated_frames}

def get_cameras(agent_dir, frame):
    imgs = glob.glob(os.path.join(agent_dir, "images", f"{frame}_*.*"))
    cams = []
    for img in imgs:
        base = os.path.basename(img)
        cam = base.split('_')[1].split('.')[0]
        cams.append((cam, img))
    return sorted(cams, key=lambda x: int(x[0]))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scenes')
def api_scenes():
    dataset = request.args.get('dataset')
    split = request.args.get('split')
    return jsonify(get_scenes(dataset, split))

@app.route('/api/frames')
def api_frames():
    dataset = request.args.get('dataset')
    split = request.args.get('split')
    scene = request.args.get('scene')
    return jsonify(get_frames(dataset, split, scene))

@app.route('/api/data')
def api_data():
    dataset = request.args.get('dataset')
    if dataset not in DATA_ROOTS:
        dataset = dataset.replace(' ', '+')
    split = request.args.get('split')
    scene = request.args.get('scene')
    frame = request.args.get('frame')

    base_dir = os.path.join(DATA_ROOTS[dataset], split)
    agent1_dir = os.path.join(base_dir, f"{scene}_1")
    agent2_dir = os.path.join(base_dir, f"{scene}_2")

    cams1 = get_cameras(agent1_dir, frame)
    cams2 = get_cameras(agent2_dir, frame)

    # Check if annotation exists in agent1's context.json
    anno_file = os.path.join(agent1_dir, "context.json")
    anno = {}
    if os.path.exists(anno_file):
        with open(anno_file, 'r') as f:
            try:
                all_anno = json.load(f)
                anno = all_anno.get(frame, {})
            except:
                pass

    return jsonify({
        'agent1': [{'cam': c[0], 'path': c[1]} for c in cams1],
        'agent2': [{'cam': c[0], 'path': c[1]} for c in cams2],
        'annotation': anno
    })

@app.route('/image')
def serve_image():
    path = request.args.get('path')
    # Decode '+' characters back if they were parsed as spaces
    path = path.replace(' ', '+')
    # Resolve the absolute path
    abs_path = os.path.abspath(path)
    directory = os.path.dirname(abs_path)
    filename = os.path.basename(abs_path)

    # We must pass the directory to send_from_directory, but it also has safety checks.
    # A more robust way to send arbitrary files outside the flask app dir is send_file:
    from flask import send_file
    try:
        return send_file(abs_path)
    except Exception as e:
        return str(e), 404

@app.route('/api/save', methods=['POST'])
def save_annotation():
    data = request.json
    dataset = data['dataset']
    if dataset not in DATA_ROOTS:
        dataset = dataset.replace(' ', '+')
    split = data['split']
    scene = data['scene']
    frame = data['frame']
    anno = data['annotation']

    base_dir = os.path.join(DATA_ROOTS[dataset], split)
    agent1_dir = os.path.join(base_dir, f"{scene}_1")
    agent2_dir = os.path.join(base_dir, f"{scene}_2")

    # We'll read the existing annotations from agent1's context.json
    anno_file = os.path.join(agent1_dir, "context.json")

    all_anno = {}
    if os.path.exists(anno_file):
        with open(anno_file, 'r') as f:
            try:
                all_anno = json.load(f)
            except:
                pass

    all_anno[frame] = anno

    # Save identical view associations to both agents' context.json files.
    if os.path.exists(agent1_dir):
        with open(os.path.join(agent1_dir, 'context.json'), 'w') as f:
            json.dump(all_anno, f, indent=4)

    if os.path.exists(agent2_dir):
        with open(os.path.join(agent2_dir, 'context.json'), 'w') as f:
            json.dump(all_anno, f, indent=4)

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
