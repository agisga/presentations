library(mvtnorm)

# function to be approximated
f <- function(x) {
  exp(cosh((x + 2*x^2 + cos(x)) / (3 + sin(x^3))))
}

# covariance function of the Gaussian process
k <- function(x, y) {
  min(x, y)
}

# evaluate the true function on a grid
x <- seq(0,1,l=100) 
fx <- f(x)
kx <- matrix(nrow = 100, ncol = 100)
for (i in 1:100) {
  for (j in 1:100) {
    kx[i,j] <- k(x[i], x[j])
  }
}

# sample paths from the prior distribution
num.samp <- 10
prior.samp <- rmvnorm(num.samp, sigma = kx)

# observe m points of the true function, then compute the posterior
m = 5
x.observed <- seq(0.1, 0.9, l = 5)
f.observed <- f(x.observed)
k.observed <- matrix(nrow = m, ncol = m)
for (i in 1:m) {
  for (j in 1:m) {
    k.observed[i,j] <- k(x.observed[i], x.observed[j])
  }
}
k.observed.vs.x <- matrix(nrow = m, ncol = 100)
for (i in 1:m) {
  for (j in 1:100) {
    k.observed.vs.x[i,j] <- k(x.observed[i], x[j])
  }
}
mu.posterior <- t(k.observed.vs.x) %*% solve(k.observed) %*% f.observed
V.posterior <- kx - t(k.observed.vs.x) %*% solve(k.observed) %*% k.observed.vs.x
posterior.samp <- rmvnorm(num.samp, mean = mu.posterior, sigma = V.posterior)

# plot the sampled paths
png("images/GP_sample_paths.png")
plot(x, fx, type = "l", ylim = c(-1,5), col = "blue", ylab = "f(x)", lwd = 2)
for (i in 1:num.samp) {
  lines(x, prior.samp[i,], type = "l", lty = 3)
  lines(x, posterior.samp[i,], lty = 2, col = "red")
}
points(x.observed, f.observed, pch = 15, cex = 1.5)
dev.off()

# now take a look at the mean
png("images/GP_posterior_mean.png")
plot(x, mu.posterior, type = "l", col = "red", ylab = "f(x)")
lines(x, fx, type = "l", col = "blue", lty = 2)
points(x.observed, f.observed, pch = 15, cex = 1.5)
dev.off()
