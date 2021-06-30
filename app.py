from flask import Flask, render_template, request, url_for, redirect, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/aboutUs', methods=['GET', 'POST'])
def aboutUs():
    if request.method == 'POST' :
        return redirect(url_for('index'))
    return render_template('aboutUs.html')

if __name__ == '__main__':
    app.run(debug=True)