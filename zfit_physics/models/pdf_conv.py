from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp
import zfit
import zfit.models.functor
from zfit import z
from zfit.exception import FunctionNotImplementedError
from zfit.util import exception, ztyping
from zfit.util.exception import WorkInProgressError


class NumConvPDFUnbinnedV1(zfit.models.functor.BaseFunctor):
    def __init__(
        self,
        func: zfit.pdf.BasePDF,
        kernel: zfit.pdf.BasePDF,
        limits: ztyping.ObsTypeInput,
        obs: ztyping.ObsTypeInput,
        ndraws: int = 20000,
        *,
        extended: ztyping.ParamTypeInput | None = None,
        name: str = "Convolution",
        experimental_pdf_normalized=False,
    ):
        """Numerical Convolution pdf of *func* convoluted with *kernel*.

        Args:
            func (:py:class:`zfit.pdf.BasePDF`): PDF  with `pdf` method that takes x and returns the function value.
                Here x is a `Data` with the obs and limits of *limits*.
            kernel (:py:class:`zfit.pdf.BasePDF`): PDF with `pdf` method that takes x acting as the kernel.
                Here x is a `Data` with the obs and limits of *limits*.
            limits (:py:class:`zfit.Space`): Limits for the numerical integration.
            obs (:py:class:`zfit.Space`): Observables of the class
            extended: If the PDF should be extended, i.e. a yield.
            ndraws (int): Number of draws for the mc integration
            name (str): Human readable name of the pdf
        """
        super().__init__(obs=obs, pdfs=[func, kernel], params={}, name=name, extended=extended)
        limits = self._check_input_limits(limits=limits)
        if limits.n_limits == 0:
            msg = "obs have to have limits to define where to integrate over."
            raise exception.LimitsNotSpecifiedError(msg)
        if limits.n_limits > 1:
            msg = "Multiple Limits not implemented"
            raise WorkInProgressError(msg)

        #        if not isinstance(func, zfit.pdf.BasePDF):
        #            raise TypeError(f"func has to be a PDF, not {type(func)}")
        #        if isinstance(kernel, zfit.pdf.BasePDF):
        #            raise TypeError(f"kernel has to be a PDF, not {type(kernel)}")

        # func = lambda x: func.unnormalized_pdf(x=x)
        # kernel = lambda x: kernel.unnormalized_pdf(x=x)

        self.conv_limits = limits
        self._ndraws = ndraws
        self._experimental_pdf_normalized = experimental_pdf_normalized

    @z.function
    def _unnormalized_pdf(self, x):
        limits = self.conv_limits
        area = limits.rect_area()[0]  # new spaces

        # create sample for numerical integral
        lower, upper = limits.rect_limits
        lower = z.convert_to_tensor(lower, dtype=self.dtype)
        upper = z.convert_to_tensor(upper, dtype=self.dtype)
        samples_normed = tfp.mcmc.sample_halton_sequence(
            dim=limits.n_obs,
            num_results=self._ndraws,
            dtype=self.dtype,
            randomized=False,
        )
        samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
        samples = zfit.Data.from_tensor(obs=limits, tensor=samples)

        func_values = self.pdfs[0].pdf(samples, norm=False)  # func of true vars

        return tf.map_fn(
            lambda xi: area * tf.reduce_mean(func_values * self.pdfs[1].pdf(xi - samples.value(), norm=False)),
            x.value(),
        )

    @zfit.supports(norm=True)
    @z.function
    def _pdf(self, x, norm):
        del norm
        if not self._experimental_pdf_normalized:
            raise FunctionNotImplementedError

        limits = self.conv_limits
        # area = limits.area()  # new spaces
        area = limits.rect_area()[0]  # new spaces

        # create sample for numerical integral
        lower, upper = limits.rect_limits
        lower = z.convert_to_tensor(lower, dtype=self.dtype)
        upper = z.convert_to_tensor(upper, dtype=self.dtype)
        samples_normed = tfp.mcmc.sample_halton_sequence(
            dim=limits.n_obs,
            num_results=self._ndraws,
            dtype=self.dtype,
            randomized=False,
        )
        samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
        samples = zfit.Data.from_tensor(obs=limits, tensor=samples)

        func_values = self.pdfs[0].pdf(samples)  # func of true vars

        return tf.map_fn(
            lambda xi: area * tf.reduce_mean(func_values * self.pdfs[1].pdf(xi - samples.value())),
            x.value(),
        )
