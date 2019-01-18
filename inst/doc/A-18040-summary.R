## ----eval=FALSE----------------------------------------------------------
#  #the use of Data frama
#  x<-c(1,1,2,2,3,3,3)
#  z<-c(80,85,92,76,61,95,83)
#  (LST<-list(class=x,sex=y,score=z))
#  LST[[3]]
#  LST[[2]][1:3]
#  LST$score
#  LST$sc
#  (student<-data.frame(x,y,z))
#  student<-data.frame(x,y,z)
#  student
#  #the use of ()
#  (student<-data.frame(class=x,sex=y,score=z))
#  student

## ----eval=FALSE----------------------------------------------------------
#  size=1;p=0.5
#  rbinom(10,size,p)
#  size=10;p=0.5
#  rbinom(20,size,p)
#  par(mfrow=c(1,3))
#  p=0.25
#  for( n in c(10,20,50))
#  { x=rbinom(100,n,p)
#  hist(x,prob=T,main=paste("n =",n))
#  xvals=0:n
#  points(xvals,dbinom(xvals,n,p),type = "h",lwd=3)
#  }
#  (par(mfrow=c(1,1)))

## ----eval=FALSE----------------------------------------------------------
#  #source("http://bioconductor.org/biocLite.R")
#  #biocLite("MASS")
#  library(MASS)
#  #install.packages("Mvnorm")
#  #library(Mvnorm)
#  sigma <- matrix(c(10,3,3,2),ncol=2)
#  x<-mvrnorm(n=500,c(1,2),sigma)
#  head(x)
#  var(x)
#  plot(x)

## ----eval=FALSE----------------------------------------------------------
#  p<-c(0.1, 0.2, 0.2, 0.2, 0.3)
#  # creat the random numbers with sample function.
#  x<-sample(0:4, size = 1000, replace = TRUE, prob = p)
#  x    #output the random numbers
#  em.p<-table(x)/1000    #calculate the empirical probabilities
#  em.th <- em.p - p     #calculate the different value of empirical and theoretical probabilities
#  rbind(em.p,p,em.th)
#  

## ----eval=FALSE----------------------------------------------------------
#  
#  beat.sl<-function(n,a,b){
#  k <- 0   #counter for accepted
#  j <- 0   #iterations
#  y <- numeric(n)
#  while (k < n) {
#  u <- runif(1)
#  j <- j + 1
#  x <- runif(1)   #random variate from g
#  if (x^(a-1) * (1-x)^(b-1) > u) {
#        #we accept x
#        k <- k + 1
#        y[k] <- x
#  } }
#  return(y)
#  }
#  
#  #par(mfrow=c(1,2))
#  
#  d<-rbeta(1000,3,2)
#  hist(d,prob = TRUE)  #output the histogram of the rbeat function in system
#  
#  c.sl<-beat.sl(1000,3,2)
#  hist(c.sl,prob = TRUE)  #output the histogram of the beat.sl function
#  
#  y <- seq(0, 1, .001)
#  i<-dbeta(y,3,2)
#  lines(y,i)  #compare the theoretical line with beat.sl function histogram
#  

## ----eval=FALSE----------------------------------------------------------
#  
#  n <- 1e3; r <- 4; beta <- 2
#  lambda <- rgamma(n, r, beta)  # creat the random numbers with rgamma function.
#  x <- rexp(n, lambda)    # creat the random numbers with rexp function.
#  options(scipen=100, digits = 3)
#  x   #output the random numbers
#  hist(x,prob = TRUE)   #output the histogram of the random numbers
#  

## ----eval=FALSE----------------------------------------------------------
#  
#  set.seed(12)
#  beta_sl<-function(x,m){
#    n<-length(x)
#    theta<-numeric(n)
#    for(i in 1:n){
#    y<-runif(m,0,x[i])   #sample Y
#    theta[i]<-mean(x[i]*(y^2)*((1-y)^2)/beta(3,3))
#  }
#    return(theta)
#  }
#  
#  x<-seq(0.1,0.9,0.1)
#  m<-100000
#  
#  x1<-beta_sl(x,m)
#  MC<-round(x1,6)  #estime result
#  pBeta<-pbeta(x,3,3)
#  
#  a<-rbind(MC,pBeta)
#  rownames(a)<-c("MC","pBeta")
#  knitr::kable(head(a),col.names=x,caption = "Comparation")   #output the result
#  
#  matplot(x,cbind(MC,pBeta),col=1:2,pch=1:2,xlim=c(0,1))   # Compare the figure
#  

## ----eval=FALSE----------------------------------------------------------
#  SL<-function(sigma,n,antithetic = TRUE){
#    u=runif(n/2)
#    if(!antithetic)
#      {v=runif(n/2)}else
#      {v=1-u}
#    u=c(u,v)
#    x=sqrt(-2*sigma^2*log(1-u))
#    return(x)
#   }
#   m<-1000
#   set.seed(123)
#   MC1<-SL(2,m)   # antithetic samples
#   MC2<-SL(2,m,antithetic = FALSE)    #independent samples
#   qqplot(MC1,MC2)
#   a=seq(0,10)
#   lines(a,a,col=2)
#  
#   b <- matrix(c(var(MC1),var(MC2),1-var(MC1)/var(MC2)),1,3)    #percent reduction in variance
#   colnames(b)<-c("Var(MC1)","Var(MC2)","1-Var(MC1)/Var(MC2)")
#   knitr::kable(head(b),digits =10)

## ----eval=FALSE----------------------------------------------------------
#  x=seq(1,5,0.1)
#  g=function(x) x^2/sqrt(2*pi)*exp(-(x^2)/2)
#  f1=function(x) exp(1-x)
#  f2=function(x) x*exp(-((x^2)-1)/2)
#  gs<-c(expression(g(x)==x^2*e^{-x^2/2}/sqrt(2*pi)),expression(f1(x)==e^(1-x)),expression(f2(x)==x*e^(-(x^2-1)/2)))
#  
#  plot(x,g(x),col=1,type="l",ylab = "",ylim = c(0,1), lwd = 0.25,main='Result')
#  points(x,f1(x),col=2,type="l",lty=1)
#  lines(x,f2(x),col=3,type="l",lty=1)
#  legend("topright",inset=.05,legend=gs,lty=1,col=1:3,horiz=FALSE)
#  

## ----eval=FALSE----------------------------------------------------------
#  
#  n=1000
#  set.seed(123)
#  X=rnorm(1.5*n,mean=1.5,sd=1)
#  X=X[X>1]
#  X=X[1:n]
#  theta1=mean(g(X)/f1(X))
#  sd1=sd(g(X)/f1(X))/n
#  
#  set.seed(123)
#  Z=rexp(n,rate=0.5)
#  Y=sqrt(Z+1)
#  theta2=mean(g(Y)/f2(Y))
#  sd2=sd(g(Y)/f2(Y))/n
#  
#   b<-matrix(c(theta1,sd1,theta2,sd2),2,2)
#   colnames(b)<-c("f1","f2")
#   rownames(b)<-c("Theta","Se")
#   knitr::kable(head(b),digits =10 ,caption = "Comparation")

## ----eval = FALSE--------------------------------------------------------
#  
#  G<-function(x){
#    n<-length(x)
#    a<-seq(1-n,n-1,2)
#    x.i<-sort(x)
#    G.hat<-sum(a*x.i)/(n*n*mean(x))
#    return(G.hat)
#  } # you can estimate a G.hat if there comes a sample
#  
#  #set.seed(1)
#  # if X is standard lognormal
#  n=500
#  m<-1000
#  G.hat1<-numeric(m)
#  for(i in 1:m){
#    x<-rlnorm(n) # then x is standard lognormal
#    G.hat1[i]<-G(x)
#  }
#  result1<-c(mean(G.hat1),quantile(G.hat1,probs=c(0.5,0.1)))
#  names(result1)<-c("mean","median","deciles")
#  print(result1)
#  hist(G.hat1,breaks=seq(min(G.hat1)-0.01,max(G.hat1)+0.01,0.01),freq =F,main = "Histogram of G",xlab = "standard lognormal")
#  
#  # if X is uniform
#  G.hat2<-numeric(m)
#  for(i in 1:m){
#    x<-runif(n) # then x is uniform
#    G.hat2[i]<-G(x)
#  }
#  result2<-c(mean(G.hat2),quantile(G.hat2,probs=c(0.5,0.1)))
#  names(result2)<-c("mean","median","deciles")
#  print(result2)
#  hist(G.hat2,breaks =seq(min(G.hat2)-0.01,max(G.hat2)+0.01,0.01) ,freq =F,main = "Histogram of G",xlab = "uniform")
#  
#  #if x is Bernoulli(0.1)
#  G.hat3<-numeric(m)
#  for(i in 1:m){
#    x<-rbinom(n,1,0.1) # then x is Bernoulli(0.1)
#    G.hat3[i]<-G(x)
#  }
#  result3<-c(mean(G.hat3),quantile(G.hat3,probs=c(0.5,0.1)))
#  names(result3)<-c("mean","median","deciles")
#  print(result3)
#  hist(G.hat3,breaks=seq(min(G.hat3)-0.01,max(G.hat3)+0.01,0.01),freq =F,main = "Histogram of G",xlab = "Bernoulli(0.1)")
#  

## ----eval=FALSE----------------------------------------------------------
#  
#  library(bootstrap) #for the law data
#  theta.hat=cor(law$LSAT, law$GPA) #we get the theta.hat
#  n=nrow(law)
#  j.cor=function(x,i)cor(x[i,1],x[i,2])
#  theta.jack=numeric(n)
#  for(i in 1:n){
#    theta.jack[i]=j.cor(law,(1:n)[-i])
#  }
#  bias.jack=(n-1)*(mean(theta.jack)-theta.hat) #we get the bias.jack
#  se.jack=sd(theta.jack)*sqrt((n-1)^2/n) #we get the se.jack
#  
#  result <- c(theta.hat,bias.jack,se.jack)
#  names(result)<-c("theta.hat","bias.jack","se.jack")
#  print(result) # print out the result.
#  

## ----eval=FALSE----------------------------------------------------------
#  library(boot) # get the air-conditioning data
#  b.mean=function(x,i)mean(x[i])
#  mean.boot=boot(data=aircondit$hours,statistic=b.mean,R=1000)
#  mean.boot
#  CI=boot.ci(mean.boot,type=c("norm","basic","perc","bca"))
#  print(CI)
#  hist(aircondit$hours,breaks = 50,freq = F)
#  

## ----eval=FALSE----------------------------------------------------------
#  library(bootstrap)  #for the scor data
#  n=nrow(scor)
#  sigma.hat=cov(scor)*(n-1)/n
#  eigenvalues.hat=eigen(sigma.hat)$values
#  theta.hat=eigenvalues.hat[1]/sum(eigenvalues.hat)
#  theta.jack=numeric(n)
#  for(i in 1:n){
#    sigma.jack=cov(scor[-i,])*(n-1)/n
#    eigenvalues.jack=eigen(sigma.jack)$values
#    theta.jack[i]=eigenvalues.jack[1]/sum(eigenvalues.jack)
#  }
#  bias.jack=(n-1)*(mean(theta.jack)-theta.hat) # get the bias.jack
#  se.jack=sd(theta.jack)*sqrt((n-1)^2/n)# get the se.jack
#  
#  result <- c(bias.jack,se.jack)
#  names(result)<-c("bias.jack","se.jack")
#  print(result) # print out the result.
#  

## ----eval=FALSE----------------------------------------------------------
#  library(DAAG)
#  attach(ironslag)
#  a <- seq(10, 40, 0.1) #sequence for plotting fits
#  
#  L1 <- lm(magnetic ~ chemical)
#  plot(chemical, magnetic, main="Linear", pch=16)
#  yhat1 <- L1$coef[1] + L1$coef[2] * a
#  lines(a, yhat1, lwd=2)
#  
#  L2 <- lm(magnetic ~ chemical + I(chemical^2))
#  plot(chemical, magnetic, main="Quadratic", pch=16)
#  yhat2 <- L2$coef[1] + L2$coef[2] * a + L2$coef[3] * a^2
#  lines(a, yhat2, lwd=2)
#  
#  L3 <- lm(log(magnetic) ~ chemical)
#  plot(chemical, magnetic, main="Exponential", pch=16)
#  logyhat3 <- L3$coef[1] + L3$coef[2] * a
#  yhat3 <- exp(logyhat3)
#  lines(a, yhat3, lwd=2)
#  
#  L4 <- lm(log(magnetic) ~ log(chemical))
#  plot(log(chemical), log(magnetic), main="Log-Log", pch=16)
#  logyhat4 <- L4$coef[1] + L4$coef[2] * log(a)
#  lines(log(a), logyhat4, lwd=2)
#  
#  n <- length(magnetic) #in DAAG ironslag
#  e1 <- e2 <- e3 <- e4 <- numeric(n/2) # for n/2-fold(leave-two-out) cross validation
#  # fit models on leave-two-out samples
#  
#  for (k in 1:(n/2)) {
#  index<-c(2*k-1,2*k)  #Subscript of leave-two-out samples point
#  y <- magnetic[-index]
#  x <- chemical[-index]
#  
#  J1 <- lm(y ~ x)
#  yhat1 <- J1$coef[1] + J1$coef[2] * chemical[index]
#  e1[k] <- mean((magnetic[index] - yhat1)^2)
#  
#  J2 <- lm(y ~ x + I(x^2))
#  yhat2 <- J2$coef[1] + J2$coef[2] * chemical[index] +
#  J2$coef[3] * chemical[index]^2
#  e2[k] <- mean((magnetic[index] - yhat2)^2)
#  
#  J3 <- lm(log(y) ~ x)
#  logyhat3 <- J3$coef[1] + J3$coef[2] * chemical[index]
#  yhat3 <- exp(logyhat3)
#  e3[k] <- mean((magnetic[index] - yhat3)^2)
#  
#  J4 <- lm(log(y) ~ log(x))
#  logyhat4 <- J4$coef[1] + J4$coef[2] * log(chemical[index])
#  yhat4 <- exp(logyhat4)
#  e4[k] <- mean((magnetic[index] - yhat4)^2)
#  }
#  
#  result <- c(mean(e1), mean(e2), mean(e3), mean(e4))
#  names(result) <- c("Lin", "Quad", "Expo", "L-L")
#  print(result) # print out the result.
#  
#  L2
#  

## ----eval=FALSE----------------------------------------------------------
#  library(latticeExtra)
#  library(RANN)
#  library(energy)
#  library(Ball)
#  library(boot)
#  library(ggplot2)

## ----eval=FALSE----------------------------------------------------------
#  set.seed(123)
#  attach(chickwts)
#  x <- sort(as.vector(weight[feed == "soybean"]))
#  y <- sort(as.vector(weight[feed == "linseed"]))
#  detach(chickwts)
#  
#  R <- 999  #number of replicates
#  z <- c(x, y)
#  K<- 1:26
#  n<-length(x)
#  reps <- numeric(R)
#  
#  CramerTwoSamples <- function(x1,x2){
#    Fx1<-ecdf(x1)
#    Fx2<-ecdf(x2)
#    n<-length(x1)
#    m<-length(x2)
#    w1<-sum((Fx1(x1)-Fx2(x1))^2)+sum((Fx1(x2)-Fx2(x2))^2)
#    w2<-w1*m*n/((m+n)^2)
#    return(w2)
#  }  #get the Cramer-von Mises statistic
#  
#  t0 <-  CramerTwoSamples (x,y)
#  for (i in 1:R) {
#    k<- sample(K, size = n, replace = FALSE)
#    x1 <- z[k]
#    y1 <- z[-k]   #complement of x1
#    reps[i] <- CramerTwoSamples (x1,y1)
#    }
#    p <- mean(abs(c(t0, reps)) >= abs(t0))
#    p
#  
#    hist(reps, main = "", freq = FALSE, xlab = "Cramer-von Mises statistic", breaks = "scott")
#    points(t0, 0, cex = 1, pch = 16)  #observed T
#  

## ----eval=FALSE----------------------------------------------------------
#  ## variable definition
#  m <- 500 #permutation samples
#  p<-2 # dimension of data
#  n1 <- n2 <- 50 #the sample size of x and y
#  R<-99 #boot parameter
#  k<-3 #boot parameter
#  n <- n1 + n2
#  N = c(n1,n2)
#  # the function of NN method
#  Tn <- function(z, ix, sizes,k){
#    n1 <- sizes[1]; n2 <- sizes[2]; n <- n1 + n2
#    if(is.vector(z)) z <- data.frame(z,0);
#    z <- z[ix, ];
#    NN <- nn2(data=z, k=k+1)
#    block1 <- NN$nn.idx[1:n1,-1]
#    block2 <- NN$nn.idx[(n1+1):n,-1]
#    i1 <- sum(block1 < n1 + .5)
#    i2 <- sum(block2 > n1+.5)
#    (i1 + i2) / (k * n)
#  }
#  
#  eqdist.nn <- function(z,sizes,k){
#    boot.obj <- boot(data=z,statistic=Tn,R=R,sim = "permutation", sizes = sizes,k=k)
#    ts <- c(boot.obj$t0,boot.obj$t)
#    p.value <- mean(ts>=ts[1])
#    list(statistic=ts[1],p.value=p.value)
#  }
#  p.values <- matrix(NA,m,3) #p
#  
#  
#  ##(1)Unequal variances and equal expectations
#  set.seed(1)
#  sd <- 1.5
#  for(i in 1:m){
#    x <- matrix(rnorm(n1*p),ncol=p)
#    y <- matrix(rnorm(n2*p,sd=sd),ncol=p)
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value#NN method
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value#energy methods
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed=i*12345)$p.value# ball method
#  }
#  alpha <- 0.05;
#  pow1 <- colMeans(p.values<alpha)
#  
#  
#  ##(2)Unequal variances and unequal expectations
#  set.seed(1)
#  mu <- 0.5
#  sd <- 1.5
#  for(i in 1:m){
#    x <- matrix(rnorm(n1*p),ncol=p)
#    y <- matrix(rnorm(n2*p,mean=mu,sd=sd),ncol=p)
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value#NN method
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value#energy methods
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed=i*12345)$p.value# ball method
#  }
#  alpha <- 0.05;
#  pow2 <- colMeans(p.values<alpha)
#  
#  
#  ##(3)Non-normal distributions: t distribution with 1 df (heavy-tailed distribution), bimodal distribution (mixture of two normal distributions)
#  set.seed(1)
#  mu <- 0.5
#  sd <- 2
#  for(i in 1:m){
#    x <- matrix(rt(n1*p,df=1),ncol=p)
#    y1 = rnorm(n2*p);  y2 = rnorm(n2*p,mean=mu,sd=sd)
#    w = rbinom(n, 1, .5) # 50:50 random choice
#    y <- matrix(w*y1 + (1-w)*y2,ncol=p)# normal mixture
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value#NN method
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value#energy methods
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed=i*12345)$p.value# ball method
#  }
#  alpha <- 0.05;
#  pow3 <- colMeans(p.values<alpha)
#  
#  
#  ##(4)Unbalanced samples
#  set.seed(1)
#  mu <- 0.5
#  N = c(n1,n2*2)
#  for(i in 1:m){
#    x <- matrix(rnorm(n1*p),ncol=p);
#    y <- cbind(rnorm(n2*2),rnorm(n2*2,mean=mu));
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value#NN method
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value#energy methods
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed=i*12345)$p.value# ball method
#  }
#  alpha <- 0.05;
#  pow4 <- colMeans(p.values<alpha)
#  
#  result <- data.frame(pow1, pow2, pow3, pow4, row.names = c('NN','energy','Ball'))
#  result
#  

## ----eval=FALSE----------------------------------------------------------
#  f <- function(x,u,lamada) {
#   return(lamada/(pi*(lamada^2+(x-u)^2)))
#  }
#  
#  m <- 500000
#  u<- 0
#  lamada <- 1
#  x <- numeric(m)
#  
#  #xt <- x[i-1]
#  #y <- rchisq(1, df = xt)
#  
#  
#  x[1] <- rnorm(1)
#  k <- 0
#  u <- runif(m)
#  for (i in 2:m) {
#    xt <- x[i-1]
#    y <- rnorm(1, mean = xt )
#    num <- f(y, 0, 1 ) * dnorm(xt, mean = y)
#    den <- f(xt, 0, 1 ) * dnorm(y, mean = xt)
#    if (u[i] <= num/den) x[i] <- y
#    else {
#           x[i] <- xt
#           k <- k+1
#           }
#  }
#  b <- 2001 #discard the burnin sample
#  y <- x[b:m]
#  a <- ppoints(100)
#  Qcauchy <- qcauchy(a)
#  Q <- quantile(x, a)
#  qqplot(Qcauchy, Q, main="",
#  xlab="Rayleigh Quantiles", ylab="Sample Quantiles", xlim=c(0,5) ,ylim=c(0,5))
#  hist(y, breaks=50, main="", xlab="", freq=FALSE)
#  lines(Qcauchy, f(Qcauchy, 0, 1))
#  

## ----eval=FALSE----------------------------------------------------------
#    w <- 0.25   #width of the uniform support set
#    m <- 5000   #length of the chain
#    burn.in <- 1000   #burn-in time
#    y <- c(125,18,20,34)
#    x <- numeric(m)   #the chain
#  
#    prob <- function(b, y) {
#       # computes (without the constant) the target density
#      if (b < 0 || b >= 1) return (0)
#      return((1/2+b/4)^y[1] * ((1-b)/4)^y[2] * ((1-b)/4)^y[3] * (b/4)^y[4])
#    }
#  
#    u <- runif(m)    #for accept/reject step
#    v <- runif(m, -w, w)    #proposal distribution
#    x[1] <-0.25
#    for (i in 2:m) {
#      z <- x[i-1] + v[i]
#      if (u[i] <= prob(z,y) / prob(x[i-1],y))
#        x[i] <-z
#      else
#        x[i] <- x[i-1]
#    }
#  
#     xb <- x[(burn.in+1):m]
#     xc<-mean(xb)
#  
#     print(xc)    #estimation value of theta
#     print(y/sum(y))
#     print(c(1/2+xc/4,(1-xc)/4,(1-xc)/4,xc/4))
#  

## ----eval=FALSE----------------------------------------------------------
#  library(latticeExtra)
#  library(RANN)
#  library(Ball)
#  library(boot)

## ----eval=FALSE----------------------------------------------------------
#  set.seed(123)
#  Gelman.Rubin <- function(psi) {
#          # psi[i,j] is the statistic psi(X[i,1:j])
#          # for chain in i-th row of X
#          psi <- as.matrix(psi)
#          n <- ncol(psi)
#          k <- nrow(psi)
#  
#          psi.means <- rowMeans(psi)     #row means
#          B <- n * var(psi.means)        #between variance est.
#          psi.w <- apply(psi, 1, "var")  #within variances
#          W <- mean(psi.w)               #within est.
#          v.hat <- W*(n-1)/n + (B/n)     #upper variance est.
#          r.hat <- v.hat / W             #G-R statistic
#          return(r.hat)
#  }
#  
#  size<-c(125,18,20,34)
#  prob <- function(y, size) {
#          # computes (without the constant) the target density
#          if (y < 0 || y >= 1)
#              return (0)
#          return((1/2+y/4)^size[1] *((1-y)/4)^size[2] * ((1-y)/4)^size[3] *(y/4)^size[4])
#  }
#  
#  thetachain<-function(m,x0){ #generate a Metropolis chain for theta
#  u <- runif(m)  #for accept/reject step
#  x<-rep(0,m)
#  x[1] <- x0
#  for (i in 2:m) {
#      y <- runif(1)
#      if (u[i] <= prob(y, size) / prob(x[i-1], size))
#          x[i] <- y  else
#     x[i] <- x[i-1]
#  }
#  return(x)
#  }
#  #generate the chains
#  k=4      #number of chains to generate
#  n=15000  #length of chains
#  b=1000   #burn-in length
#  x0=c(0.3,0.4,0.5,0.6)
#  theta <- matrix(0, nrow=k, ncol=n)
#  for (i in 1:k)
#    theta[i, ] <- thetachain(m=n,x0=x0[i])
#  #compute diagnostic statistics
#  psi <- t(apply(theta, 1, cumsum))
#  for (i in 1:nrow(psi))
#    psi[i,] <- psi[i,] / (1:ncol(psi))
#  print(Gelman.Rubin(psi))
#  #plot psi for the four chains
#  par(mfrow=c(2,2))
#  for (i in 1:k)
#  plot(psi[i, (b+1):n], type="l",xlab=i, ylab=bquote(psi))
#  par(mfrow=c(1,1)) #restore default
#  
#  #the sequence of R-hat statistics
#  rhat <- rep(0, n)
#  for (j in (b+1):n)
#  rhat[j] <- Gelman.Rubin(psi[,1:j])
#  plot(rhat[(b+1):n], type="l", xlab="", ylab="R")
#  abline(h=1.2, lty=2)
#  

## ----eval=FALSE----------------------------------------------------------
#  f <- function(x, theta, eta) {
#    #sita mean scale parameter
#    #eta mean the location parameter
#  1/(theta*pi*(1+((x-eta)/theta)^2))
#  }
#  
#  up<- seq(-20, 20, 4) #intergrand upper bound
#  v <- rep(0, length(up))
#  for (i in 1:length(up)) {
#  #standard Cauchy distribution with theta=1,eta=0
#  v[i]<-integrate(f,lower=-Inf,upper=up[i],rel.tol=.Machine$double.eps^0.25,theta=1,eta=0)$value
#  }
#  
#  data.frame(x=up,estamator=v,rcauchy.est=pcauchy(up),error=v-pcauchy(up))
#  

## ----eval=FALSE----------------------------------------------------------
#  
#  # Mle function
#  eval_f0 <- function(x,x1,n.A=28,n.B=24,nOO=41,nAB=70) {
#    #x[1] mean p , x1[1] mean p0
#    #x[2] mean q , x1[2] mean q0
#    r1<-1-sum(x1)
#    nAA<-n.A*x1[1]^2/(x1[1]^2+2*x1[1]*r1)
#    nBB<-n.B*x1[2]^2/(x1[2]^2+2*x1[2]*r1)
#    r<-1-sum(x)
#    return(-2*nAA*log(x[1])-2*nBB*log(x[2])-2*nOO*log(r)-
#             (n.A-nAA)*log(2*x[1]*r)-(n.B-nBB)*log(2*x[2]*r)-nAB*log(2*x[1]*x[2]))
#  }
#  
#  
#  # constraint function
#  eval_g0 <- function(x,x1,n.A=28,n.B=24,nOO=41,nAB=70) {
#    return(sum(x)-0.999999)
#  }
#  
#  opts <- list("algorithm"="NLOPT_LN_COBYLA",
#               "xtol_rel"=1.0e-8)
#  mle<-NULL
#  r<-matrix(0,1,2)
#  r<-rbind(r,c(0.2,0.35))# the beginning value of p0 and q0
#  j<-2
#  while (sum(abs(r[j,]-r[j-1,]))>1e-8) {
#  res <- nloptr( x0=c(0.3,0.25),
#                 eval_f=eval_f0,
#                 lb = c(0,0), ub = c(1,1),
#                 eval_g_ineq = eval_g0,
#                 opts = opts, x1=r[j,],n.A=28,n.B=24,nOO=41,nAB=70 )
#  j<-j+1
#  r<-rbind(r,res$solution)
#  mle<-c(mle,eval_f0(x=r[j,],x1=r[j-1,]))
#  }
#  r  #the result of EM algorithm
#  mle #the max likelihood values
#  

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  attach(mtcars)
#  formulas <- list( mpg ~ disp, mpg ~ I(1 / disp), mpg ~ disp + wt, mpg ~ I(1 / disp) + wt )
#  #for loops
#  out <- vector("list", length(formulas))
#  for (i in seq_along(formulas)) { out[[i]] <-lm(formulas[[i]]) }
#  out
#  
#  #lapply
#  lapply(formulas,lm)
#  

## ----eval=FALSE----------------------------------------------------------
#  
#  bootstraps <- lapply(1:10, function(i) {
#    rows <- sample(1:nrow(mtcars), rep = TRUE)
#    mtcars[rows, ] })
#  
#  #for loops
#  for(i in seq_along(bootstraps)){
#    print(lm(mpg~disp,data =bootstraps[[i]]))
#  }
#  
#  #lapply
#  lapply(bootstraps,lm,formula=mpg~disp)
#  

## ----eval=FALSE----------------------------------------------------------
#  #in exercise 3
#  rsq <- function(mod) summary.lm(mod)$r.squared
#  #for loops
#  for (i in seq_along(formulas)) {
#   print( rsq(lm(formulas[[i]])))
#    }
#  #lapply
#  lapply(lapply(formulas,lm),rsq)
#  #in exercise 4
#  #for loops
#  for(i in seq_along(bootstraps)){
#    print(rsq(lm(mpg~disp,data =bootstraps[[i]])))
#  }
#  
#  #lapply
#  lapply(lapply(bootstraps,lm,formula=mpg~disp),rsq)

## ----eval=FALSE----------------------------------------------------------
#  #using anonymous function
#  trials <- replicate( 100, t.test(rpois(10, 10), rpois(7, 10)), simplify = FALSE )
#  p_value<-function(mod) mod$p.value
#  sapply(trials, p_value)
#  

## ----eval=FALSE----------------------------------------------------------
#  x <- list(a = c(1:3), b = c(4:8))
#  
#  lapply.f<-function(x,f,...){
#     r<-Map(f,x,...)
#     n<-length(r[[1]])
#     return(vapply(r,as.vector,numeric(n)))
#  }
#  lapply.f(x,mean)
#  lapply.f(x,quantile)
#  

## ----eval=FALSE----------------------------------------------------------
#  library(microbenchmark)
#  library(latticeExtra)
#  library(Ball)
#  library(nloptr)

## ----eval=FALSE----------------------------------------------------------
#  
#  my.chisq.test<-function(x,y){
#  if(!is.vector(x) && !is.vector(y))
#  stop("at least one of 'x' and 'y' is not a vector")
#  if(typeof(x)=="character" || typeof(y)=="character")
#  stop("at least one of 'x' and 'y' is not a numeric vector")
#  if(any(x<0) || anyNA(x))
#  stop("all entries of 'x' must be nonnegative and finite")
#  if(any(y<0) || anyNA(y))
#  stop("all entries of 'y' must be nonnegative and finite")
#  if((n<-sum(x))==0)
#  stop("at least one entry of 'x' must be positive")
#  if((n<-sum(x))==0)
#  stop("at least one entry of 'x' must be positive")
#  if(length(x)!=length(y))
#  stop("'x' and 'y' must have the same length")
#  DNAME<-paste(deparse(substitute(x)),"and",deparse(substitute(y)))
#  METHOD<-"Pearson's Chi-squared test"
#  x<-rbind(x,y)
#  nr<-as.integer(nrow(x));nc<-as.integer(ncol(x))
#  sr<-rowSums(x);sc<-colSums(x);n<-sum(x)
#  E<-outer(sr,sc,"*")/n
#  STATISTIC<-sum((x - E)^2/E)
#  names(STATISTIC)<-"X-squared"
#  structure(list(statistic=STATISTIC,method=METHOD,data.name=DNAME),class="htest")
#  }
#  
#  #There is an example.
#  mya<-c(762,327,468);myb<-c(484,239,477)
#  my.chisq.test(mya,myb)
#  chisq.test(rbind(mya,myb))
#  microbenchmark(t1=my.chisq.test(mya,myb),t2=chisq.test(rbind(mya,myb)))
#  

## ----eval=FALSE----------------------------------------------------------
#  
#  my.table<-function(...,dnn = list.names(...),deparse.level = 1){
#      list.names <- function(...) {
#          l <- as.list(substitute(list(...)))[-1L]
#          nm <- names(l)
#          fixup <- if (is.null(nm))
#              seq_along(l)
#          else nm == ""
#          dep <- vapply(l[fixup], function(x) switch(deparse.level +
#              1, "", if (is.symbol(x)) as.character(x) else "",
#              deparse(x, nlines = 1)[1L]), "")
#          if (is.null(nm))
#              dep
#          else {
#              nm[fixup] <- dep
#              nm
#          }
#      }
#      args <- list(...)
#      if (!length(args))
#          stop("nothing to tabulate")
#      if (length(args) == 1L && is.list(args[[1L]])) {
#          args <- args[[1L]]
#          if (length(dnn) != length(args))
#              dnn <- if (!is.null(argn <- names(args)))
#                  argn
#              else paste(dnn[1L], seq_along(args), sep = ".")
#      }
#      bin <- 0L
#      lens <- NULL
#      dims <- integer()
#      pd <- 1L
#      dn <- NULL
#      for (a in args) {
#          if (is.null(lens))
#              lens <- length(a)
#          else if (length(a) != lens)
#              stop("all arguments must have the same length")
#          fact.a <- is.factor(a)
#          if (!fact.a) {
#              a0 <- a
#              a <- factor(a)
#          }
#          ll <- levels(a)
#          a <- as.integer(a)
#          nl <- length(ll)
#          dims <- c(dims, nl)
#          dn <- c(dn, list(ll))
#          bin <- bin + pd * (a - 1L)
#          pd <- pd * nl
#      }
#      names(dn) <- dnn
#      bin <- bin[!is.na(bin)]
#      if (length(bin))
#          bin <- bin + 1L
#      y <- array(tabulate(bin, pd), dims, dimnames = dn)
#      class(y) <- "table"
#      y
#  }
#  
#  #There is an example.
#  mya<-myb<-c(1,seq(1,4))
#  my.table(mya,myb)
#  table(mya,myb)
#  microbenchmark(t1=my.table(mya,myb),t2=table(mya,myb))
#  

