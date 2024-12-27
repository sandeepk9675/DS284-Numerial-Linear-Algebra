import numpy as np
import matplotlib.pyplot as plt
import csv

img = plt.imread('input_image.png')
print(img.shape)

U_0, S_0, V_0 = np.linalg.svd(img[:,:,0])
print(U_0.shape, S_0.shape, V_0.shape)
U_1, S_1, V_1 = np.linalg.svd(img[:,:,1])
print(U_1.shape, S_1.shape, V_1.shape)
U_2, S_2, V_2 = np.linalg.svd(img[:,:,2])
print(U_2.shape, S_2.shape, V_2.shape)



# Perform SVD on each color channel with economy decomposition
U_0, S_0, V_0 = np.linalg.svd(img[:, :, 0], full_matrices=False)
U_1, S_1, V_1 = np.linalg.svd(img[:, :, 1], full_matrices=False)
U_2, S_2, V_2 = np.linalg.svd(img[:, :, 2], full_matrices=False)


# Print shapes of the SVD components
print(f"Channel 0 SVD shapes: U={U_0.shape}, S={S_0.shape}, V={V_0.shape}")
print(f"Channel 1 SVD shapes: U={U_1.shape}, S={S_1.shape}, V={V_1.shape}")
print(f"Channel 2 SVD shapes: U={U_2.shape}, S={S_2.shape}, V={V_2.shape}")

# Initialize error list
error_frobenius = []
error_two_norm = []
error_frobenius_RGB_theo = []
error_two_norm_RGB_theo = []
error_frobenius_RGB_exp = []
error_two_norm_RGB_exp = []

# Define the list of singular values to use, ensuring they don't exceed the maximum
rank = [5, 10, 20, 40, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]

size_plot = len(rank)
# Set up the plot for reconstructed images
fig, ax1 = plt.subplots(3, 5, figsize=(20, 8))

n = 0

# Determine if the image uses floating point or integer types
if img.dtype in [np.float32, np.float64]:
    is_float = True
    print("Image is in floating-point format.")
else:
    is_float = False
    print("Image is in integer format.")

for singular_values in rank:
    # Reconstruct each channel using the specified number of singular values
    img_r_channel_0 = np.dot(U_0[:, :singular_values],
                             np.dot(np.diag(S_0[:singular_values]), V_0[:singular_values, :]))
    img_r_channel_1 = np.dot(U_1[:, :singular_values],
                             np.dot(np.diag(S_1[:singular_values]), V_1[:singular_values, :]))
    img_r_channel_2 = np.dot(U_2[:, :singular_values],
                             np.dot(np.diag(S_2[:singular_values]), V_2[:singular_values, :]))
    
    # Combine the channels into a single image
    img_r = np.zeros(img.shape, dtype=img.dtype)
    img_r[:, :, 0] = img_r_channel_0
    img_r[:, :, 1] = img_r_channel_1
    img_r[:, :, 2] = img_r_channel_2

    # Clip the reconstructed image to the valid range
    if is_float:
        img_r_clipped = np.clip(img_r, 0, 1)
    else:
        img_r_clipped = np.clip(img_r, 0, 255)

    # Calculate the Frobenius norm of the difference (reconstruction error)
    temp_error_in_channel_0_f = np.linalg.norm(img[:, :, 0] - img_r[:, :, 0], ord='fro')
    temp_error_in_channel_1_f = np.linalg.norm(img[:, :, 1] - img_r[:, :, 1], ord='fro')
    temp_error_in_channel_2_f = np.linalg.norm(img[:, :, 2] - img_r[:, :, 2], ord='fro')
    overall_error_frobenius = np.sqrt(temp_error_in_channel_0_f**2 + temp_error_in_channel_1_f**2 + temp_error_in_channel_2_f**2)
    error_frobenius.append(overall_error_frobenius)
    error_frobenius_RGB_exp.append([temp_error_in_channel_0_f, temp_error_in_channel_1_f, temp_error_in_channel_2_f])

    # Theoretical Frobenius error calculation
    temp_error_in_channel_0_f_theo = np.sqrt(np.sum((S_0[singular_values:])**2))
    temp_error_in_channel_1_f_theo = np.sqrt(np.sum((S_1[singular_values:])**2))
    temp_error_in_channel_2_f_theo = np.sqrt(np.sum((S_2[singular_values:])**2))
    error_frobenius_RGB_theo.append([temp_error_in_channel_0_f_theo, temp_error_in_channel_1_f_theo, temp_error_in_channel_2_f_theo])

    # Calculate the 2-norm of the difference (reconstruction error)
    temp_error_in_channel_0_t = np.linalg.norm(img[:, :, 0] - img_r[:, :, 0], ord=2)
    temp_error_in_channel_1_t = np.linalg.norm(img[:, :, 1] - img_r[:, :, 1], ord=2)
    temp_error_in_channel_2_t = np.linalg.norm(img[:, :, 2] - img_r[:, :, 2], ord=2)
    overall_error_two_norm = np.sqrt(temp_error_in_channel_0_t**2 + temp_error_in_channel_1_t**2 + temp_error_in_channel_2_t**2)
    error_two_norm.append(overall_error_two_norm)
    error_two_norm_RGB_exp.append([temp_error_in_channel_0_t, temp_error_in_channel_1_t, temp_error_in_channel_2_t])

    # Theoretical 2-norm error calculation
    temp_error_in_channel_0_t_theo = S_0[singular_values] if singular_values < len(S_0) else 0
    temp_error_in_channel_1_t_theo = S_1[singular_values] if singular_values < len(S_1) else 0
    temp_error_in_channel_2_t_theo = S_2[singular_values] if singular_values < len(S_2) else 0
    error_two_norm_RGB_theo.append([temp_error_in_channel_0_t_theo, temp_error_in_channel_1_t_theo, temp_error_in_channel_2_t_theo])

    # Display the reconstructed image
    col = n % 5
    row = n // 5
    ax1[row, col].imshow(img_r_clipped)
    ax1[row, col].set_title(f"Rank {singular_values}")
    ax1[row, col].axis('off')
    n += 1

# Adjust layout
plt.tight_layout()
plt.show()

plt.plot(rank, error_frobenius, label="Frobenius norm", marker='o')
plt.plot(rank, error_two_norm, label="2-norm", marker='+')
plt.xlabel('Rank')
plt.ylabel('Reconstruction error')
plt.legend()  # Add the legend here
plt.savefig('reconstruction_error.png')
plt.show()  # Optional, to display the plot

# File name for the CSV
output_file = 'error_metricspy.csv'

# Write data to CSV file
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers
    writer.writerow([
        "Frobenius RGB Theo - R", "Frobenius RGB Theo - G", "Frobenius RGB Theo - B",
        "2-norm RGB Theo - R", "2-norm RGB Theo - G", "2-norm RGB Theo - B",
        "Frobenius RGB Exp - R", "Frobenius RGB Exp - G", "Frobenius RGB Exp - B",
        "2-norm RGB Exp - R", "2-norm RGB Exp - G", "2-norm RGB Exp - B"
    ])
    
    # Write the data row by row
    for i in range(len(error_frobenius_RGB_theo)):
        # Fetching data from each list
        row = (
            error_frobenius_RGB_theo[i] +
            error_two_norm_RGB_theo[i] +
            error_frobenius_RGB_exp[i] +
            error_two_norm_RGB_exp[i]
        )
        writer.writerow(row)

print(f"Data has been written to {output_file}")