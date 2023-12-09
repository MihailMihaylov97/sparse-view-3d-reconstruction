import torch
import numpy as np

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def rot3d_to_2d_viewing(look_at):

    # Extract the camera's forward vector (negative z-axis of the camera's coordinate system)
    forward_vector = -look_at[:, 2]

    # Calculate the elevation angle (vertical angle)
    elevation = np.arcsin(forward_vector[1])

    # Calculate the azimuth angle (horizontal angle)
    if np.abs(np.cos(elevation)) > 1e-6:
        azimuth = np.arctan2(forward_vector[0], forward_vector[2])
    else:
        # Handle edge case when elevation is close to 90 degrees
        azimuth = 0.0  # Define a default value for azimuth

    return torch.from_numpy(np.array(azimuth[None])), torch.from_numpy(np.array(elevation[None]))


def extract_ph_theta(centers):
    radius = torch.norm(centers, p=2, dim=-1)
    theta = torch.acos(centers[:, 1] / radius)

    phi = torch.atan2(
        centers[:, 0] / (radius * torch.cos(theta)),
        centers[:, 2] / (radius * torch.cos(theta)),
    )
    theta = theta * 180 / np.pi
    phi = phi * 180 / np.pi

    return theta, phi, radius




def get_poses(poses):
    poses_1 = torch.FloatTensor([[ 6.8935126e-01,5.3373039e-01,-4.8982298e-01,-1.9745398e+00],
                                [-7.2442728e-01,5.0788772e-01,-4.6610624e-01,-1.8789345e+00],
                                [ 1.4901163e-08,6.7615211e-01,7.3676193e-01,2.9699826e+00],
                                [ 0.0000000e+00,0.0000000e+00,0.0000000e+00,1.0000000e+00]])


    poses_2 = torch.FloatTensor([[ 0.08438807,0.5158344,-0.85252184,-3.4366255 ],
                                [-0.99643296,0.0436861,-0.07220022,-0.29104838],
                                [ 0.,0.8555737,0.51768094,2.0868387],
                                [ 0.,0.,0.,1.]])
    
    to_opencv = torch.FloatTensor([[1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,1,0],
                                    [0,0,0,1]])
    
    poses_1 = to_opencv @ poses_1
    poses_2 = to_opencv @ poses_2
    
    
    
    poses[0,...] = poses_1
    poses[1,...] = poses_2
    return poses 


def get_view_direction(thetas, phis, overhead, front):
    #                   phis: [B,];          thetas: [B,]
    # front = 0             [-front/2, front/2)
    # side (cam left) = 1   [front/2, 180-front/2)
    # back = 2              [180-front/2, 180+front/2)
    # side (cam right) = 3  [180+front/2, 360-front/2)
    # top = 4               [0, overhead]
    # bottom = 5            [180-overhead, 180]
    print(thetas, phis, overhead, front)
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


    
def get_phi_theta(poses):
    azi, ele = [], []
    for pose in poses:
        azimuth, elevation = rot3d_to_2d_viewing(pose.squeeze().cpu().numpy())
        azimuth += np.pi
        elevation += np.pi/2
        
        azi.append(azimuth[None])
        ele.append(elevation[None])
    
    phis = torch.from_numpy(np.array(azi)).to(pose.device).float().squeeze()
    thetas = torch.from_numpy(np.array(ele)).to(pose.device).float().squeeze() 
    return phis, thetas



def extract_spherical(cam2world):
    R = cam2world[:, :3, :3]
    T = cam2world[:, :3, 3]
    
    r = torch.norm(T, dim=-1)
    azimuth = torch.arctan2(R[:, 1,0], R[:, 0,0])
    elevation = torch.arctan2(-R[:,2,0], torch.sqrt(R[:,2,1]**2 + R[:,2,2]**2))
    
    azimuth = torch.rad2deg(azimuth)
    elevation = torch.rad2deg(elevation)
    return azimuth, elevation, r
    
    
def circle_poses(device, radius, theta, phi, return_dirs, angle_overhead, angle_front, generate_pose):
    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
        radius * torch.sin(theta) * torch.cos(phi),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)

    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = torch.cross(forward_vector, right_vector, dim=-1)
    

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(len(centers), 1, 1)
    if generate_pose: 
        poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
        poses[:, :3, 3] = centers

    else:
        poses = get_poses(poses)
    

    
    phi, theta, radius = extract_spherical(poses)
    print(f"Calculated these sphericals: {phi=} {theta=} {radius=}")
    
    # phi, theta, radius = extract_ph_theta(poses[:1,:3,3])
    # print(f"{phi=} {theta=} {radius=}")
    
    # phi, theta = get_phi_theta(poses)
    
    if return_dirs:
        dirs = get_view_direction(torch.deg2rad(theta), torch.deg2rad(phi), angle_overhead, angle_front)
    else:
        dirs = None

    # phi = phi / np.pi * 180
    # theta = theta / np.pi * 180

    return poses, dirs, phi, theta

    
if __name__=="__main__":
    
    theta_range = [117.7818,117.7818]
    phi_range = [326.3828,326.3828]
    radius_range = [4.031129,4.031129]


    # theta_range = [-8.537738277053595e-07, 94.1404]
    # phi_range = [-46.4212237985909, 301.2675]
    # theta_range = [27.7818,  4.1404]
    # phi_range = [146.3828, 121.2676]

    # phi_range = [-33.6172, -58.7325]
    # theta_range = [-27.7818,  -4.1404]



    # theta_range = [60,80]
    # phi_range = [60,80]
    # radius_range = [4,4]
    
    # theta_range = [117.7818,117.7818]
    # phi_range = [326.3828,326.3828]
    # radius_range = [4.031129,4.031129]

    
    angle_overhead = 30
    angle_front = 60 
    return_dirs = True
    device="cpu"

    # rand_pose(theta_range, phi_range, radius_range, angle_overhead, angle_front)


    generate_pose = False
    poses, dirs, phi, theta = circle_poses(device, 
                                           radius=torch.tensor(radius_range), 
                                           theta=torch.tensor(theta_range), 
                                           phi=torch.tensor(phi_range), 
                                           return_dirs=False, 
                                           angle_overhead=angle_overhead, 
                                           angle_front=angle_front, 
                                           generate_pose=generate_pose)
    

    print(f"{poses=} {dirs=} {phi=} {theta=}")



    poses_2, dirs, phi_2, theta_2 = circle_poses(device, 
                                           radius=torch.tensor(radius_range), 
                                           theta=theta, 
                                           phi=phi, 
                                           return_dirs=False, 
                                           angle_overhead=angle_overhead, 
                                           angle_front=angle_front, 
                                           generate_pose=True)
    
    
    print(f"{poses_2=} {dirs=} {phi_2=} {theta_2=}")
    
    
    print(torch.isclose(poses, poses_2))


    
    
    