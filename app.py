from flask import Flask, render_template, redirect, url_for
# from subprocess import call

app=Flask(__name__)

@app.route('/')
def welcome():
    return redirect(url_for('index'), code=302)


@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/entertainment')
def entertainment():
    # camera.process_img
    return render_template("Entertainment.html")

@app.route('/food')
def food():
    # camera.process_img
    return render_template("food.html")

@app.route('/messages')
def messages():
    # camera.process_img
    return render_template("messages.html")

@app.route('/keyboard')
def keyboard():
    # camera.process_img
    return render_template("keyboard.html")

@app.route('/medicine')
def medicine():
    # camera.process_img
    return render_template("medicine.html")

@app.route('/normal')
def normal():
    # camera.process_img
    return render_template("normal.html")

@app.route('/quick_message')
def quick_message():
    # camera.process_img
    return render_template("quick_message.html")

@app.route('/water')
def water():
    # camera.process_img()
    return render_template("water.html")

app.run()