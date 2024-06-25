from builtins import super
import numpy as np
import numbers
from .emregistration import EMRegistration
from .utility import is_positive_semi_definite

class RigidRegistration(EMRegistration):
    def __init__(self, R=None, t=None, s=None, scale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.D != 2 and self.D != 3:
            raise ValueError('Rigid registration only supports 2D or 3D point clouds. Instead got {}.'.format(self.D))
        if R is not None and ((R.ndim != 2) or (R.shape[0] != self.D) or (R.shape[1] != self.D) or not is_positive_semi_definite(R)):
            raise ValueError('The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(self.D, self.D, R))
        if t is not None and ((t.ndim != 2) or (t.shape[0] != 1) or (t.shape[1] != self.D)):
            raise ValueError('The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(self.D, t))
        if s is not None and (not isinstance(s, numbers.Number) or s <= 0):
            raise ValueError('The scale factor must be a positive number. Instead got: {}.'.format(s))

        self.R = np.eye(self.D) if R is None else R
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t
        self.s = 1 if s is None else s
        self.scale = scale

    def update_transform(self):
        muX = np.divide(np.sum(self.PX, axis=0), self.Np)
        muY = np.divide(np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)
        self.X_hat = self.X - np.tile(muX, (self.N, 1))
        Y_hat = self.Y - np.tile(muY, (self.M, 1))
        self.YPY = np.dot(np.transpose(self.P1), np.sum(np.multiply(Y_hat, Y_hat), axis=1))
        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D, ))
        C[self.D-1] = np.linalg.det(np.dot(U, V))
        self.R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
        if self.scale is True:
            self.s = np.trace(np.dot(np.transpose(self.A), np.transpose(self.R))) / self.YPY
        self.t = np.transpose(muX) - self.s * np.dot(np.transpose(self.R), np.transpose(muY))

    def transform_point_cloud(self, Y=None):
        if Y is None:
            self.TY = self.s * np.dot(self.Y, self.R) + self.t
            return
        else:
            return self.s * np.dot(Y, self.R) + self.t

    def update_variance(self):
        qprev = self.q
        trAR = np.trace(np.dot(self.A, self.R))
        xPx = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X_hat, self.X_hat), axis=1))
        self.q = (xPx - 2 * self.s * trAR + self.s * self.s * self.YPY) / (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
        self.diff = np.abs(self.q - qprev)
        self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        return self.s, self.R, self.t
