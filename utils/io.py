import numpy as np
import os


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension == '.npy':
            return cls._read_npy(file_path)
        elif file_extension == '.xyz':
            return cls._read_xyz(file_path)
        elif file_extension == '.txt':
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def put(cls, file_path, *args, **kwargs):
        _, file_extension = os.path.splitext(file_path)
        dir_name = os.path.dirname(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if file_extension == '.xyz':
            return cls._write_xyz(file_path, *args, **kwargs)
        elif file_extension == '.npy':
            return cls._write_npy(file_path, *args, **kwargs)
        elif file_extension == '.ply':
            return cls._write_ply(file_path, *args, **kwargs)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path).astype(np.float32)

    @classmethod
    def _read_xyz(cls, file_path):
        return np.loadtxt(file_path).astype(np.float32)

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _write_xyz(cls, file_path, pcd):
        np.savetxt(file_path, pcd)

    @classmethod
    def _write_npy(cls, file_path, pcd):
        np.save(file_path, pcd)

    @classmethod
    def _write_ply(cls, file_path, pcd, color_params=None):

        with open(file_path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('comment PCL generated\n')
            f.write('element vertex {}\n'.format(len(pcd)))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')

            if color_params is None:
                y = pcd[:, 2:3].copy()
                y -= y.min()
                y /= y.max()
                bottom_color = np.tile(
                    np.array([[0.0, 0.0, 1.0]]), (y.shape[0], 1))
                top_color = np.tile(
                    np.array([[0.0, 1.0, 0.0]]), (y.shape[0], 1))
                color = (1-y)*bottom_color + y*top_color
            else:
                color = np.tile(
                    np.array([[0.3, 0.3, 0.3]]), (pcd.shape[0], 1))
                for color_param in color_params:
                    mask, c = color_param
                    assert mask.shape == (pcd.shape[0], )
                    r, g, b = c[0], c[1], c[2]
                    color[mask] = np.array([r, g, b])

            for i in range(len(pcd)):
                r, g, b = color[i][0], color[i][1], color[i][2]
                x, y, z = pcd[i]
                f.write('{} {} {} {} {} {}\n'.format(
                    x, y, z, int(255*r), np.int(255*g), int(255*b)))
