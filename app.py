from flask import Flask, render_template, request, url_for, redirect, render_template

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST' :
        return redirect(url_for('index'))
    
    return render_template('contact.html')

if __name__ == '__main__':
    application.run(debug=True)