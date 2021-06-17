from flask import Flask, request, url_for, redirect, render_template

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/blog', methods=['GET', 'POST'])
def blog():
    if request.method == 'POST' :
        return redirect(url_for('index'))
    return render_template('blog.html')

@application.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST' :
        return redirect(url_for('index'))
    return render_template('contact.html')

@application.route('/blog-detail', methods=['GET', 'POST'])
def blog_detail():
    if request.method == 'POST' :
        return redirect(url_for('index'))
    return render_template('blog-detail.html')

@application.route('/project-detail', methods=['GET', 'POST'])
def project_detail():
    if request.method == 'POST' :
        return redirect(url_for('index'))
    return render_template('project-detail.html')

if __name__ == '__main__':
    application.run(debug=True)