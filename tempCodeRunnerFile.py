predictions = model.predict(test_images)

# print("I think its", CLASS_NAMES[np.argmax(predictions[0])])
# show_img(arr[-1])