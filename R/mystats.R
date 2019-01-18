# This is an example function named 'mystats'
# which calculates the descriptive statistics.

#' @title mystats
#' @description an example function calculating the descriptive statistics
#' @param x the data vector or matrix
#' @param na.omit the logistic value
#' @return a random sample of size \code{n}
#' @examples

#' myvars <- c("mpg","hp","wt")
#' sapply(mtcars[myvars],mystats)
#' @export
mystats <- function(x,na.omit=FALSE) {
 if(na.omit)
   x<-x[!is.na(x)]
 m<-mean(x)
 n<-length(x)
 s<-sd(x)
 skew<-sum((x-m)^3/s^3)/n
 kurt<-sum((x-m)^4)/s^4/n-3
 return(c(n=n,mean=m,stdev=s,skew=skew,kurtosis=kurt))
}
