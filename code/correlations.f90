module correlation_functions
  implicit none

  public

  real(8), allocatable, dimension(:) :: k, Pk, k2Pk
  real(8), parameter :: pi = atan(1d0) * 4d0
  real(8), parameter :: twopi2 = 2 * pi**2

contains

  subroutine set_k(kval, Pkval, n)
    real(8), dimension(1:n), intent(in) :: kval, Pkval
    integer, intent(in) :: n

    if (allocated(k)) deallocate(k, Pk, k2Pk)
    allocate(k(1:n), Pk(1:n), k2Pk(1:n))

    k = kval
    Pk = Pkval
    k2Pk = k**2 * Pk
  end subroutine set_k

  elemental real(8) function j0(x)
    real(8), intent(in) :: x
    real(8) :: x2
    if (x < 1d-4) then
       x2 = x**2
       j0 = (1d0-7d0*x2/60d0) / (1d0+x2/20d0)
    else
       j0 = sin(x) / x
    end if
  end function j0

  elemental real(8) function j1(x)
    real(8), intent(in) :: x
    real(8) :: x2

    if (x < 1d-4) then
       x2 = x**2
       j1 = x / (3d0 + 3d0 * x2 / 10d0)
    else
       j1 = (sin(x)/x - cos(x)) / x
    end if
  end function j1

  elemental real(8) function j2(x)
    real(8), intent(in) :: x
    real(8) :: x2, x4

    x2 = x**2
    if (x < 1d-3) then
       x4 = x**4
       j2 =(x2 / 15d0 - 181d0 * x4 / 76230d0 ) / (1d0 + 13d0 * x2 / 363d0 + 5d0 * x4 / 8712d0)
    else
       j2 = (3d0/x2 - 1) * sin(x) / x - 3d0 * cos(x) / x2
    end if
  end function j2

  elemental real(8) function j3(x)
    real(8), intent(in) :: x
    real(8) :: x2, x3, x4

    x2 = x**2
    x3 = x**3
    x4 = x**4
    ! Use Pade approximant of order 5
    if (x < 1d-1) then
       j3 = (x**3/105d0 - (79*x**5)/319410d0)/(1 + (5*x**2)/169d0 + (17*x**4)/44616d0)
    else
       j3 = (15d0/x4 - 6d0/x2) * sin(x) - (15d0 / x2 - 1d0) * cos(x) / x
    end if
  end function j3

  function besselj(n, x, M) result (res)
    integer, intent(in) :: n, M
    real(8), intent(in), dimension(M) :: x

    real(8), dimension(M) :: res

    if (n == 0) then
       res = j0(x)
    else if (n == 1) then
       res = j1(x)
    else if (n == 2) then
       res = j2(x)
    else if (n == 3) then
       res = j3(x)
    else
       res = -1d99
    end if
  end function besselj

  elemental real(8) function W(x)
    real(8), intent(in) :: x
    W = exp(-x**2 / 2)
  end function W

  function trapz(A, x, n) result(res)
    real(8), dimension(1:n), intent(in) :: A, x
    integer, intent(in) :: n

    real(8) :: res
    integer :: i

    res = 0
    do i = 1, n-1
       res = res + (x(i+1) - x(i)) * (A(i+1) + A(i)) / 2
    end do
  end function trapz

  subroutine xi(n, m, R1, R2, r, nR, res)
    integer, intent(in) :: n, m, nR
    real(8), intent(in) :: R1, R2
    real(8), intent(in), dimension(1:nR) :: r

    real(8), intent(out), dimension(1:nR) :: res

    real(8), dimension(1:size(k)) :: tmp, integrand, kr
    integer :: i, nk

    nk = size(k)
    integrand = k2Pk * W(k * R1) * W(k * R2) / (twopi2)

    do i = 1, nR
       kr = k * r(i)
       if (m == 0) then
          tmp = integrand * besselj(n, kr, nk)
       else
          tmp = integrand * besselj(n, kr, nk) / kr**m
       end if

       res(i) = trapz(tmp, k, size(k))
    end do
  end subroutine xi

  subroutine chi(n, m, R1, R2, r, nR, res)
    integer, intent(in) :: n, m, nR
    real(8), intent(in) :: R1, R2
    real(8), intent(in), dimension(1:nR) :: r

    real(8), intent(out), dimension(1:nR) :: res
    call xi(n, -m, R1, R2, r, nR, res)
  end subroutine chi

end module correlation_functions
