import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pylab

# 画图
fig = plt.figure()
ims = []
for i in range(3):
    im = pylab.plot(
        i * 4,
        i * 4,
        color="#ff0000",
        marker="D",
    )
    im.extend(plt.plot(
        i * 2,
        i + 7,
        color="#00ff00",
        marker="1",
    ))

    ims.append(im)

ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=10000)
ani.save("test1.gif", writer='pillow')