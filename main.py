import numpy as np
import cv2
import argparse as ap
import imageio
from tqdm import tqdm
from scipy.interpolate import interpn
from scipy.signal import convolve2d
from utils import lRGB2XYZ, XYZ2lRGB, xyY_to_XYZ, gamma_correction_sRGB
from skimage.util import compare_images
import matplotlib.pyplot as plt
from pathlib import Path


def read_im(image_path):
    return cv2.imread(image_path, -1)


def get_intensity_influence(x, sigma=0.05):
    return np.exp(-np.square(x) / (2 * (sigma**2)))


def bilateral_filter(ambient_im,
                     flash_im=None,
                     spatial_sigma=2,
                     intensity_sigma=0.05):
    """
    Assumes ambient_im, flash_im are in [0, 255]
    
    """
    l = 0.01
    # ambient_im = ambient_im / (2**16 - 1)
    ambient_im = ambient_im / 255
    if flash_im is None:
        fname = f"bilateral_filtering_spatial_{spatial_sigma}_intensity_{intensity_sigma}.png"
        flash_im = ambient_im
        minI = ambient_im.min() - l
        maxI = ambient_im.max() + l
    else:
        fname = f"joint_bilateral_filtering_spatial_{spatial_sigma}_intensity_{intensity_sigma}.png"
        flash_im = flash_im / 255
        minI = flash_im.min() - l
        maxI = flash_im.max() + l

    NB_SEGMENTS = int(np.ceil((maxI - minI) / intensity_sigma))
    J = [[], [], []]
    I_amb = ambient_im
    I_flash = flash_im
    for j in tqdm(range(NB_SEGMENTS)):
        for i in range(3):
            ij = minI + j * (maxI - minI) / NB_SEGMENTS
            Gj = get_intensity_influence((I_flash[:, :, i] - ij),
                                         sigma=intensity_sigma)
            Kj = cv2.GaussianBlur(Gj, ksize=(0, 0), sigmaX=spatial_sigma)
            Hj = Gj * I_amb[:, :, i]
            Hstarj = cv2.GaussianBlur(Hj, ksize=(0, 0), sigmaX=spatial_sigma)
            Jj = Hstarj / Kj
            J[i].append(Jj)

    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(I_amb.shape[1]), np.arange(I_amb.shape[0]),
        minI + np.arange(NB_SEGMENTS) * (maxI - minI) / NB_SEGMENTS)

    grid = (np.arange(I_amb.shape[0]), np.arange(I_amb.shape[1]),
            np.linspace(minI, maxI, NB_SEGMENTS))
    values = [np.dstack(J[i]) for i in range(3)]
    image_grids = [
        np.dstack([y_coords[..., 0], x_coords[..., 0], I_amb[:, :, i]])
        for i in range(3)
    ]
    image = [interpn(grid, values[i], image_grids[i]) for i in tqdm(range(3))]
    image = np.dstack(image)

    return image


def detail_transfer(A_NR, F, spatial_sigma=1, intensity_sigma=0.05):
    """
    Assumes A_NR is in range [0, 1]
    Assumes F is in range [0, 255]
    """
    # cv2.imwrite("joint_bilateral.png", A_NR * 255)
    F_Base = bilateral_filter(F,
                              F,
                              spatial_sigma=spatial_sigma,
                              intensity_sigma=intensity_sigma)
    F = F / 255
    eps = 1e-8
    A_Detail = A_NR * (F + eps) / (F_Base + eps)
    return np.clip(A_Detail, 0.0, 1.0)


def masking(A,
            F,
            A_Detail,
            A_Base,
            tau_shadow=0.1,
            ambient_iso=1600,
            flash_iso=200):
    """
    Assumes A is in range [0, 255]
    Assume F is in range [0, 255]
    Assumes A_Detail is in range [0, 1]
    Assumes A_Base is in range [0, 1]
    lamp_ambient ISO: 1600
    lamp_flash ISO: 200
    
    """
    A = A / 255
    F = F / 255
    A_lin = gamma_correction_sRGB(A) * (flash_iso / ambient_iso)
    F_lin = gamma_correction_sRGB(F)

    A_XYZ = lRGB2XYZ(A_lin)
    A_Y = A_XYZ[..., 1]

    F_XYZ = lRGB2XYZ(F_lin)
    F_Y = F_XYZ[..., 1]

    M_shad = np.zeros_like(A_Y)
    M_spec = np.zeros_like(A_Y)
    M_shad[np.where((F_Y - A_Y) <= tau_shadow)] = 1

    sensor_outputs = np.sort(np.unique(F_Y.reshape(1, -1)))
    spec_threshold = sensor_outputs[int(0.95 * len(sensor_outputs))]
    M_spec[np.where(F_Y >= spec_threshold)] = 1

    M = M_shad + M_spec
    M[np.nonzero(M)] = 1
    M = np.expand_dims(M, axis=-1)

    A_Final = (1 - M) * A_Detail + M * A_Base
    return A_Final


def gradient(img):
    """
    Input: H x W input in [0, 1] range
    Output: H x W x 2 output Ix, Iy
    """
    imgx = np.pad(img, ((0, 0), (0, 1)), mode="constant")
    imgy = np.pad(img, ((0, 1), (0, 0)), mode="constant")
    Ix = np.diff(imgx, n=1, axis=1)
    Iy = np.diff(imgy, n=1, axis=0)
    return np.dstack([Ix, Iy])


def divergence(gradient_field):
    """
    Input: gradient field H x W x 2
    Output: H x W
    """
    gradX, gradY = gradient_field[..., 0], gradient_field[..., 1]
    gradient_field_X = np.pad(gradX, ((0, 0), (1, 0)), mode="constant")
    gradient_field_Y = np.pad(gradY, ((1, 0), (0, 0)), mode="constant")
    gradXX = np.diff(gradient_field_X, n=1, axis=1)
    gradYY = np.diff(gradient_field_Y, n=1, axis=0)

    return gradXX + gradYY


def laplacian(img):
    """
    Input: H x W in [0, 1] range
    Output: H x W
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    if len(img.shape) == 3:
        out = np.dstack([
            convolve2d(img[..., i],
                       kernel,
                       mode="same",
                       boundary="fill",
                       fillvalue=0) for i in range(3)
        ])
    else:
        out = convolve2d(img,
                         kernel,
                         mode="same",
                         boundary="fill",
                         fillvalue=0)
    return out


def test_gradient_divergence_laplacian(img):
    i1 = divergence(gradient(img))
    i2 = laplacian(img)
    assert (i1[1:-1, 1:-1] - i2[1:-1, 1:-1]).sum() == 0


def poisson_solver(init_image, boundary_img, divergence, eps=1e-4, N=10000):
    I_star_init = init_image.copy()
    B = np.zeros_like(boundary_img)
    boundary = 1
    B[boundary:-boundary, boundary:-boundary] = 1
    I_star_boundary = boundary_img.copy()
    I_star_boundary[boundary:-boundary, boundary:-boundary] = 0

    I_star = B * I_star_init + (1 - B) * I_star_boundary
    r = B * (divergence - laplacian(I_star))
    d = r
    delta_new = (r * r).sum()
    n = 0

    while np.sqrt((r * r).sum()) > eps and n < N:
        q = laplacian(d)
        eta = delta_new / ((d * q).sum())
        I_star = I_star + B * (eta * d)
        r = B * (r - eta * q)
        delta_old = delta_new
        delta_new = (r * r).sum()
        beta = delta_new / delta_old
        d = r + beta * d
        n += 1
        print(f"Iteration: {n}, Eta: {eta}, Value: {np.sqrt((r*r).sum()) }")

    return I_star


def fused_gradient_field(ambient_im, flash_im):
    """
    Assumes ambient_im, flash_im in [0, 255]
    """
    ambient_im = ambient_im / 255
    flash_im = flash_im / 255
    grad_a = gradient(ambient_im)
    grad_phi_prime = gradient(flash_im)
    M_numerator = (grad_phi_prime * grad_a).sum(axis=2)
    M_numerator = np.abs(M_numerator)

    M_denominator = np.sqrt(np.square(grad_phi_prime).sum(axis=2)) * np.sqrt(
        np.square(grad_a).sum(axis=2))
    M_denominator += 1e-8

    M = M_numerator / M_denominator
    M = np.expand_dims(M, -1)
    sigma = 40
    tau_s = 0.9
    ws = np.tanh(sigma * (flash_im - tau_s))
    ws = (ws - ws.min()) / (ws.max() - ws.min())
    ws = np.expand_dims(ws, -1)
    grad_phi_star = (ws * grad_a) + (1 - ws) * (M * grad_phi_prime +
                                                (1 - M) * grad_a)

    div_phi_star = divergence(grad_phi_star)

    return grad_phi_star, div_phi_star


def reflection_removal(ambient_im, flash_im, tau_ue=0.9):
    ambient_im = ambient_im / 255
    flash_im = flash_im / 255
    print("Ambient: ", ambient_im.sum())
    print("Flash: ", flash_im.sum())
    H = ambient_im + flash_im
    grad_H = gradient(H)
    print("Grad H: ", grad_H.sum())
    grad_a = gradient(ambient_im)
    print("Grad A: ", grad_a.sum())
    proj_H_a = grad_a * np.expand_dims(np.abs(
        (grad_H * grad_a).sum(axis=2)), -1) / (np.square(
            np.linalg.norm(grad_a, axis=(0, 1))))
    print("Proj: ", proj_H_a.sum())
    # return
    sigma = 40
    # breakpoint()
    wue = 1 - np.tanh(sigma * (flash_im - tau_ue))
    print("W: ", wue.sum())
    wue = (wue - wue.min()) / (wue.max() - wue.min())
    wue = np.expand_dims(wue, -1)
    grad_phi_star = (wue * grad_H) + (1 - wue) * proj_H_a
    print("Grad phi: ", grad_phi_star.sum())
    div_phi_star = divergence(grad_phi_star)
    print("Divergence: ", div_phi_star.sum())
    return div_phi_star


def main(args):
    # return
    ambient_im = read_im(args.ambient)
    flash_im = read_im(args.flash)
    spatial_sigma = args.sigma_s
    intensity_sigma = args.sigma_r
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
    if args.bilateral:
        A_Base = bilateral_filter(ambient_im, None, spatial_sigma,
                                  intensity_sigma)
        cv2.imwrite(
            str(out_dir / f"bilateral_{spatial_sigma}_{intensity_sigma}.png"),
            np.clip(A_Base, 0, 1) * 255)
    if args.joint:
        A_NR = bilateral_filter(ambient_im, flash_im, spatial_sigma,
                                intensity_sigma)
        cv2.imwrite(
            str(out_dir / f"joint_{spatial_sigma}_{intensity_sigma}.png"),
            np.clip(A_NR, 0, 1) * 255)
    if args.detail:
        A_NR = bilateral_filter(ambient_im, flash_im, spatial_sigma,
                                intensity_sigma)
        A_Detail = detail_transfer(A_NR, flash_im, spatial_sigma,
                                   intensity_sigma)
        cv2.imwrite(
            str(out_dir / f"detail_{spatial_sigma}_{intensity_sigma}.png"),
            np.clip(A_Detail, 0, 1) * 255)
    if args.masked:
        A_Base = bilateral_filter(ambient_im, None, spatial_sigma,
                                  intensity_sigma)
        A_NR = bilateral_filter(ambient_im, flash_im, spatial_sigma,
                                intensity_sigma)
        A_Detail = detail_transfer(A_NR, flash_im, spatial_sigma,
                                   intensity_sigma)
        A_Final = masking(ambient_im,
                          flash_im,
                          A_Detail,
                          A_Base,
                          tau_shadow=args.tau_shadow,
                          ambient_iso=args.ambient_iso,
                          flash_iso=args.flash_iso)
        cv2.imwrite(
            str(out_dir / f"final_{spatial_sigma}_{intensity_sigma}.png"),
            np.clip(A_Final, 0, 1) * 255)

    if args.gradient_domain:
        ambient_im = cv2.resize(
            ambient_im, (ambient_im.shape[1] // 4, ambient_im.shape[0] // 4))
        flash_im = cv2.resize(flash_im,
                              (flash_im.shape[1] // 4, flash_im.shape[0] // 4))
        target_divergence = np.dstack([
            fused_gradient_field(ambient_im[..., i], flash_im[..., i])[1]
            for i in range(3)
        ])
        ims = {
            "zero": np.zeros_like(ambient_im[..., :3]),
            "ambient": ambient_im[..., :3] / 255.0,
            "flash": flash_im / 255.0,
            "average": (ambient_im[..., :3] + flash_im[..., :3]) / (2 * 255.0)
        }
        img = poisson_solver(ims[args.init], ims[args.boundary],
                             target_divergence, args.eps, args.num_iters)
        cv2.imwrite(
            str(out_dir /
                f"gradient_domain_init_{args.init}_boundary_{args.boundary}.png"
                ),
            np.clip(img, 0, 1) * 255)

    if args.get_gradients:
        im_names = ["ambient", "flash", "fused"]
        ims = [
            ambient_im / 255, flash_im / 255,
            np.dstack([
                fused_gradient_field(ambient_im[..., i], flash_im[..., i])[0]
                for i in range(3)
            ])
        ]
        for im_name, im in zip(im_names, ims):
            for i, channel in enumerate(['r', 'g', 'b']):
                print(im_name, channel)
                if im_name == "fused":
                    grad = im[..., (i * 2):((i * 2) + 2)]
                else:
                    grad = gradient(im[..., i])
                plt.imshow(grad[..., 0], cmap='gray')
                plt.axis('off')
                plt.savefig(str(out_dir / f"grad_{im_name}_{channel}_x.jpeg"),
                            bbox_inches="tight",
                            pad_inches=0,
                            dpi=200)
                plt.imshow(grad[..., 1], cmap='gray')
                plt.axis('off')
                plt.savefig(str(out_dir / f"grad_{im_name}_{channel}_y.jpeg"),
                            bbox_inches="tight",
                            pad_inches=0,
                            dpi=200)
    if args.reflection_removal:
        ambient_im = cv2.resize(
            ambient_im, (ambient_im.shape[1] // 4, ambient_im.shape[0] // 4))
        flash_im = cv2.resize(flash_im,
                              (flash_im.shape[1] // 4, flash_im.shape[0] // 4))
        ims = {
            "zero": np.zeros_like(ambient_im[..., :3]),
            "ambient": ambient_im[..., :3] / 255.0,
            "flash": flash_im / 255.0,
            "average": (ambient_im[..., :3] + flash_im[..., :3]) / (2 * 255.0)
        }
        target_divergence = np.dstack([
            reflection_removal(ambient_im[..., i],
                               flash_im[..., i],
                               tau_ue=1.0) for i in range(3)
        ])
        img = poisson_solver(ims[args.init], ims[args.boundary],
                             target_divergence, args.eps, args.num_iters)

        cv2.imwrite(
            str(out_dir /
                f"reflection_removed_init_{args.init}_boundary_{args.boundary}.png"
                ),
            np.clip(img, 0, 1) * 255)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--ambient",
                        help="Path to ambient image",
                        required=True)
    parser.add_argument("--flash", help="Path to flash image", required=True)
    parser.add_argument("--out_dir",
                        help="Path to write results",
                        default="./out_images/")
    parser.add_argument("--sigma_r", default=2, type=float, required=False)
    parser.add_argument("--sigma_s", default=0.1, type=float, required=False)
    parser.add_argument("--bilateral",
                        action=ap.BooleanOptionalAction,
                        required=False,
                        default=False)
    parser.add_argument("--joint",
                        action=ap.BooleanOptionalAction,
                        required=False,
                        default=False)
    parser.add_argument("--detail",
                        action=ap.BooleanOptionalAction,
                        required=False,
                        default=False)
    parser.add_argument("--masked",
                        action=ap.BooleanOptionalAction,
                        required=False,
                        default=False)
    parser.add_argument("--tau_shadow",
                        default=0.1,
                        type=float,
                        required=False)
    parser.add_argument("--ambient_iso",
                        default=1600,
                        type=int,
                        required=False)
    parser.add_argument("--flash_iso", default=200, type=int, required=False)
    parser.add_argument("--gradient_domain",
                        action=ap.BooleanOptionalAction,
                        default=False,
                        required=False)
    parser.add_argument("--eps", default=1e-4, type=float, required=False)
    parser.add_argument("--num_iters", default=10000, type=int, required=False)
    parser.add_argument("--init",
                        default="zero",
                        type=str,
                        required=False,
                        help="One of [zero, ambient, flash, average]")
    parser.add_argument("--boundary",
                        default="zero",
                        type=str,
                        required=False,
                        help="One of [zero, ambient, flash, average]")
    parser.add_argument("--get_gradients",
                        action=ap.BooleanOptionalAction,
                        default=False,
                        required=False)
    parser.add_argument("--reflection_removal",
                        action=ap.BooleanOptionalAction,
                        default=False,
                        required=False)
    args = parser.parse_args()
    for a in dir(args):
        print(f"--{a}: (Optional)")
    main(args)