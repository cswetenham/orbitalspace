#ifndef FIXED64_H
#define	FIXED64_H

double const half_scale = (1ULL<<32);
double const fractional_scale = half_scale * half_scale;

class Fixed64
{
public:
  Fixed64(Fixed64 const& other) :
    integral_part(other.integral_part),
    fractional_part(other.fractional_part)
  {}
  Fixed64() :
    integral_part(0),
    fractional_part(0)
  {}
  Fixed64(int64_t integral, int64_t fractional) :
    integral_part(integral),
    fractional_part(fractional)
  { printBinary(); }
  static Fixed64 fromDouble(double v) {
    // both of these are negative if v is negative
    double integral_part_d;
    double fractional_part_d = fabs(modf(v, &integral_part_d));

    int64_t integral_part = (int64_t)integral_part_d;
    uint64_t fractional_part = (uint64_t)(fractional_part_d * fractional_scale);

    if (v < 0.0 && fractional_part_d != 0.0) {
      // borrow from integral part
      integral_part = (int64_t)integral_part_d - 1;
      // TODO what happens if I remove the abs() above and just do fractional_part = (uint64_t)(fractional_part_d * fractional_scale) ?
      fractional_part = ~((uint64_t)(fractional_part_d * fractional_scale)) + 1;
    }
    printf("fromDouble id: %e fd: %e ii: %ld fi: %ld\n", integral_part_d, fractional_part_d, integral_part, fractional_part);
    return Fixed64(integral_part, fractional_part);
  }
  static double toDouble(Fixed64 v) {
    double integral_part_d = (double)v.integral_part;
    double fractional_part_d = (double)v.fractional_part / fractional_scale;
    printf("toDouble ii: %ld fi: %ld id: %e fd: %e\n", v.integral_part, v.fractional_part, integral_part_d, fractional_part_d);
    return integral_part_d + fractional_part_d;
  }
  void printBinary() const {
    std::stringstream ss;
    for (int i = 63; i >= 0; --i) {
      ss << ((((uint64_t)integral_part) & (1ULL << i)) ? "1" : "0");
    }
    ss << ".";
    for (int i = 63; i >= 0; --i) {
      ss << (((fractional_part) & (1ULL << i)) ? "1" : "0");
    }
    printf("%s\n", ss.str().c_str());
  }
  Fixed64 operator+(Fixed64 const& rhs) {
    int64_t carry = (fractional_part & rhs.fractional_part) >> 63;
    return Fixed64(integral_part + rhs.integral_part + carry, fractional_part + rhs.fractional_part);
  }
private:
   int64_t integral_part;
  uint64_t fractional_part;
};

void assert_equal(char const* msg, double t, double u) {
  if (t != u) {
    printf("Assertion failed: %s: saw values %e and %e (difference: %e).\n", msg, t, u, t-u);
    assert(false);
  }
}

#define ASSERT_EQ(LHS, RHS) do { assert_equal(#LHS " == " #RHS, LHS, RHS); } while (0)

void run_tests() {
  ASSERT_EQ(half_scale / (1UL << 32), 1.0);
  ASSERT_EQ((fractional_scale / (1UL << 32)) / (1UL << 32), 1.0);
  // Small positive integers
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(0.0)), 0.0);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(1.0)), 1.0);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(2.0)), 2.0);
  // Small negative integers
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(-1.0)), -1.0);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(-2.0)), -2.0);
  // Small positive values
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(0.5)), 0.5);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(-0.5)), -0.5);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(-0.125)), -0.125);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(-1.625)), -1.625);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(-0.625)), -0.625);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(-2.625)), -2.625);

  // Positive ints
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(1.0) + Fixed64::fromDouble(0.0)), 1.0);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(1.0) + Fixed64::fromDouble(1.0)), 2.0);
  // Negative ints
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(-2.0) + Fixed64::fromDouble(-1.0)), -3.0);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(1.0) + Fixed64::fromDouble(-1.0)), 0.0);
  // Fractions
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(0.5) + Fixed64::fromDouble(0.5)), 1.0);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(1.5) + Fixed64::fromDouble(1.5)), 3.0);
  // Negative fractions
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(-0.5) + Fixed64::fromDouble(-0.5)), -1.0);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(-0.25) + Fixed64::fromDouble(0.25)), 0.0);
  ASSERT_EQ(Fixed64::toDouble(Fixed64::fromDouble(-2.625) + Fixed64::fromDouble(2.625)), 0.0);

  printf("All tests passed.\n");
}

#endif	/* FIXED64_H */

