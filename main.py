import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from numba import njit # Для компиляции в машинный код


@njit(cache=True)
def diamond_square(n_iter, roughness=0.5):
    size = 2 ** n_iter + 1
    terrain = np.zeros((size, size))
    # Угловые точки получают случайные значения от -1 до 1.
    terrain[0::size - 1, 0::size - 1] = np.random.uniform(-1, 1, (2, 2))

    step_size = size - 1
    current_roughness = roughness

    for _ in range(n_iter):
        half = step_size // 2

        # Diamond step
        for x in range(0, size - step_size, step_size):
            for y in range(0, size - step_size, step_size):
                # Это среднее значение четырёх углов квадрата, плюс немного случайности.
                # Это создаёт новые точки по рёбрам и соединяет ландшафт.
                avg = (terrain[x, y] + terrain[x + step_size, y] +
                       terrain[x, y + step_size] + terrain[x + step_size, y + step_size]) / 4
                # устанавливает значение в центральной точке квадрата на основе среднего значения углов и шума
                terrain[x + half, y + half] = avg + np.random.uniform(-1, 1) * current_roughness

        # Square step
        for x in range(0, size, half):
            """
            Берутся 2–4 соседние точки, вычисляется среднее, добавляется шум.
            Это заполняет промежуточные точки и сглаживает карту.
            """
            for y in range((x + half) % step_size, size, step_size):
                total = 0.0
                count = 0
                if x >= half:
                    total += terrain[x - half, y]
                    count += 1
                if x + half < size:
                    total += terrain[x + half, y]
                    count += 1
                if y >= half:
                    total += terrain[x, y - half]
                    count += 1
                if y + half < size:
                    total += terrain[x, y + half]
                    count += 1
                if count > 0:
                    terrain[x, y] = total / count + np.random.uniform(-1, 1) * current_roughness

        step_size = half
        current_roughness *= 0.5
    return terrain


@njit(cache=True)
def progressive_generate(base, new_iter, roughness):
    current_size = base.shape[0]
    current_iter = int(np.log2(current_size - 1)) # Вычисляется текущий n по размеру
    new_size = 2 ** new_iter + 1

    terrain = np.zeros((new_size, new_size))
    step = (new_size - 1) // (current_size - 1)

    """
    каждая точка из старой карты копируется с шагом step (чтобы оставить место для новых точек между ними). 
    Эти пустые места будут потом заполнены с помощью алгоритма.
     |
    """
    terrain[::step, ::step] = base

    for iter_num in range(current_iter, new_iter):
        step_size = 2 ** (new_iter - iter_num)
        half = step_size // 2
        current_rough = roughness * (0.5 ** iter_num)

        # Diamond step
        for x in range(0, new_size - step_size, step_size):
            for y in range(0, new_size - step_size, step_size):
                # Это среднее значение четырёх углов квадрата, плюс немного случайности.
                avg = (terrain[x, y] + terrain[x + step_size, y] +
                       terrain[x, y + step_size] + terrain[x + step_size, y + step_size]) / 4
                terrain[x + half, y + half] = avg + np.random.uniform(-1, 1) * current_rough

        # Square step
        """
        Берутся 2–4 соседние точки, вычисляется среднее, добавляется шум.
        Это заполняет промежуточные точки и сглаживает карту.
        """
        for x in range(0, new_size, half):
            for y in range((x + half) % step_size, new_size, step_size):
                total = 0.0
                count = 0
                if x >= half:
                    total += terrain[x - half, y]
                    count += 1
                if x + half < new_size:
                    total += terrain[x + half, y]
                    count += 1
                if y >= half:
                    total += terrain[x, y - half]
                    count += 1
                if y + half < new_size:
                    total += terrain[x, y + half]
                    count += 1
                if count > 0:
                    terrain[x, y] = total / count + np.random.uniform(-1, 1) * current_rough

    return terrain


def show():
    plt.show()


class DiamondSquareApp:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(11, 12))

        # Оставляем место справа под элементы управления
        plt.subplots_adjust(left=0.05, right=0.7, bottom=0.1, top=0.95)

        self.initial_n = 5
        self.initial_r = 0.5
        self.curr_n = self.initial_n
        self.curr_r = self.initial_r

        self.terrain = diamond_square(self.curr_n, self.curr_r)

        """
        Показывает 2D массив как изображение.
        Значения в массиве (высоты) сопоставляются с цветами из палитры.
        """
        self.img = self.ax.imshow(
            self.terrain, cmap='terrain',
            extent=[0, self.terrain.shape[1], 0, self.terrain.shape[0]]
        )
        self.ax.set_title('Diamond-Square')
        self.ax.set_aspect('equal')

        # === Слайдер N ===
        self.ax_n = self.fig.add_axes([0.75, 0.65, 0.2, 0.03])
        self.n_slider = Slider(self.ax_n, 'N (iterations)', 1, 10, valinit=self.curr_n, valstep=1)

        # === Слайдер R ===
        self.ax_r = self.fig.add_axes([0.75, 0.6, 0.2, 0.03])
        self.r_slider = Slider(self.ax_r, 'R (roughness)', 0.1, 2.0, valinit=self.curr_r, valstep=0.1)

        # === Кнопка Reset ===
        self.reset_ax = self.fig.add_axes([0.75, 0.5, 0.1, 0.05])
        self.reset_button = Button(self.reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset)

        # === Кнопка Save ===
        self.save_ax = self.fig.add_axes([0.85, 0.5, 0.1, 0.05])
        self.save_button = Button(self.save_ax, 'Save')
        self.save_button.on_clicked(self.save)

        # Подключаем обработчики
        self.n_slider.on_changed(self.update_iter)
        self.r_slider.on_changed(self.update_rough)

    def reset(self, event):
        """Сброс к исходным параметрам"""
        self.n_slider.set_val(self.initial_n)
        self.r_slider.set_val(self.initial_r)

        self.curr_n = self.initial_n
        self.curr_r = self.initial_r
        self.terrain = diamond_square(self.curr_n, self.curr_r)
        self.update_image()

    def save(self, event):
        """Сохранение текущего сценария фрактала"""
        filename = f"diamond-square_with_{self.curr_n}_{self.curr_r:.2f}.jpg"
        plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpeg')
        print(f"Изображение сохранено как {filename}")

    def update_terrain(self):
        """Обновляем ландшафт на основе текущих параметров"""
        new_n = self.n_slider.val
        new_r = self.r_slider.val

        if new_n > self.curr_n:
            self.terrain = progressive_generate(self.terrain, new_n, new_r)
        else:
            self.terrain = diamond_square(new_n, new_r)

        self.curr_n = new_n
        self.curr_r = new_r
        self.update_image()

    def update_image(self):
        """Обновление изображения без повторной генерации карты"""
        self.img.set_data(self.terrain)
        self.img.set_extent([0, self.terrain.shape[1], 0, self.terrain.shape[0]])
        self.img.set_clim(self.terrain.min(), self.terrain.max())
        self.ax.set_xlim(0, self.terrain.shape[1])
        self.ax.set_ylim(0, self.terrain.shape[0])
        self.fig.canvas.draw_idle()

    def update_iter(self, val):
        """Обработчик изменения детализации"""
        self.update_terrain()

    def update_rough(self, val):
        """Обработчик изменения шероховатости"""
        self.update_terrain()


if __name__ == "__main__":
    app = DiamondSquareApp()
    show()