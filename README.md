# Traffic Sign Classifier - CS50 AI

## My Experimentation Process
I tested different CNN architectures to find the best model for traffic sign classification.
At first, I used only two convolutional layers, but the accuracy was pretty low (~60%).
Then, I added more layers and experimented with dropout to prevent overfitting.

## What Worked Well
- Adding more convolutional layers improved accuracy, especially using 32, 64, and 128 filters.
- Dropout (`0.5`) helped reduce overfitting.
- The Adam optimizer was fast and provided good results.

## What Didn't Work Well
- Without dropout, the model overfitted and performed worse on test data.
- Using fewer filters resulted in much lower accuracy.
- Adding too many layers slowed down training without significant improvements.

## Final Model
In the end, I chose a model with **3 convolutional layers, max-pooling, dropout, and a dense layer**.
The final model achieved **~95-97% accuracy** on test data, which is quite good!

## Conclusion
I learned a lot about CNNs and deep learning while working on this project.
It was interesting to see how small changes could affect accuracy.
Further improvements could be made by experimenting with data augmentation or different optimizers.
