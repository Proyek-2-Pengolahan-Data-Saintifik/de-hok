from flask import Flask, render_template, request, url_for, redirect, render_template

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/aboutUs', methods=['GET', 'POST'])
def aboutUs():
    if request.method == 'POST' :
        return redirect(url_for('index'))
    return render_template('aboutUs.html')

if __name__ == '__main__':
    application.run(debug=True)